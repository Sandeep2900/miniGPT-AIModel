$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\data')).Path
$files = Get-ChildItem -LiteralPath $root -Recurse -File | Sort-Object FullName
$textExt = @('.txt','.json','.jsonl','.csv','.tsv','.md')

function Mask-Value([string]$value) {
    $s = ($value -replace '\s+', ' ').Trim()
    if ($s.Length -le 4) { return ('*' * $s.Length) }
    return $s.Substring(0,2) + ('X' * [Math]::Max(2, $s.Length - 4)) + $s.Substring($s.Length - 2)
}

Write-Output '=== INVENTORY: relative path | bytes | line count (text files only) ==='
foreach ($f in $files) {
    $relative = $f.FullName.Substring($root.Length + 1)
    if ($textExt -contains $f.Extension.ToLowerInvariant()) {
        $lineCount = (Get-Content -LiteralPath $f.FullName -Encoding UTF8 | Measure-Object -Line).Lines
    } else { $lineCount = 'N/A (binary/non-text)' }
    Write-Output "$relative | $($f.Length) | $lineCount"
}

Write-Output '=== PRIVACY SCAN: counts and first 3 masked samples per pattern ==='
$patterns = [ordered]@{
    phone = '(?<!\d)(?:\+?\d[\d\s().-]{8,}\d)(?!\d)'
    email = '(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b'
    url = '(?i)\b(?:https?://|www\.)[^\s<>"''{}|\\^\[\]]+'
    secret_keyword = '(?i)\b(?:password|passwd|pin|otp)\b'
}
foreach ($f in $files) {
    if ($textExt -notcontains $f.Extension.ToLowerInvariant()) { continue }
    $text = Get-Content -LiteralPath $f.FullName -Raw -Encoding UTF8
    $relative = $f.FullName.Substring($root.Length + 1)
    foreach ($key in $patterns.Keys) {
        $found = [regex]::Matches($text, $patterns[$key])
        $samples = @($found | Select-Object -First 3 | ForEach-Object { Mask-Value $_.Value })
        Write-Output "$relative | $key=$($found.Count) | samples=$($samples -join '; ')"
    }
}

Write-Output '=== INTENTS JSONL VALIDATION ==='
foreach ($f in ($files | Where-Object { $_.Name -like 'intents*.jsonl' })) {
    $total = 0; $badJson = 0; $missingFields = 0; $emptyReplies = 0
    foreach ($line in Get-Content -LiteralPath $f.FullName -Encoding UTF8) {
        if (-not $line.Trim()) { continue }
        $total++
        try { $obj = $line | ConvertFrom-Json -ErrorAction Stop } catch { $badJson++; continue }
        $missing = $false
        foreach ($field in @('intent','phrase','replies')) {
            if ($null -eq $obj.PSObject.Properties[$field]) { $missing = $true; break }
        }
        if ($missing) { $missingFields++ }
        if ($null -ne $obj.PSObject.Properties['replies'] -and ($null -eq $obj.replies -or @($obj.replies).Count -lt 1)) { $emptyReplies++ }
    }
    Write-Output "$($f.Name) | records=$total invalid_json=$badJson missing_required_fields=$missingFields empty_replies=$emptyReplies"
}

Write-Output '=== TOOL FILE FORMAT; first five nonblank unrecognized lines ==='
foreach ($name in @('tool_train.txt','tool_val.txt')) {
    $path = Join-Path $root $name
    if (-not (Test-Path -LiteralPath $path)) { Write-Output "$name | missing"; continue }
    $userCount = 0; $assistantCount = 0; $timeCount = 0; $badCount = 0; $badLines = @()
    foreach ($line in Get-Content -LiteralPath $path -Encoding UTF8) {
        if ($line.StartsWith('User:')) { $userCount++ }
        elseif ($line.StartsWith('Assistant:')) { $assistantCount++ }
        elseif ($line.StartsWith('Current time:')) { $timeCount++ }
        elseif ($line.Trim()) { $badCount++; if ($badLines.Count -lt 5) { $badLines += ,$line } }
    }
    Write-Output "$name | User=$userCount Assistant=$assistantCount CurrentTime=$timeCount nonblank_other=$badCount"
    foreach ($badLine in $badLines) { Write-Output "BAD | $badLine" }
}

Write-Output '=== TRAIN PAIRING AND UNCLASSIFIED LINES ==='
$trainPath = Join-Path $root 'train.txt'
if (Test-Path -LiteralPath $trainPath) {
    $lines = @(Get-Content -LiteralPath $trainPath -Encoding UTF8)
    $userCount = 0; $assistantCount = 0; $otherCount = 0; $pairCount = 0
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i].StartsWith('User:')) {
            $userCount++
            if ($i + 1 -lt $lines.Count -and $lines[$i + 1].StartsWith('Assistant:')) { $pairCount++ }
        } elseif ($lines[$i].StartsWith('Assistant:')) { $assistantCount++ }
        elseif ($lines[$i].Trim()) { $otherCount++ }
    }
    Write-Output "User=$userCount Assistant=$assistantCount adjacent_User_then_Assistant=$pairCount nonblank_neither=$otherCount"
}

Write-Output '=== CHARACTER MIX; non-whitespace Unicode code points ==='
foreach ($f in $files) {
    if ($textExt -notcontains $f.Extension.ToLowerInvariant()) { continue }
    $text = Get-Content -LiteralPath $f.FullName -Raw -Encoding UTF8
    $dev = [regex]::Matches($text, '[\u0900-\u097F]').Count
    $latin = [regex]::Matches($text, '[A-Za-z]').Count
    $nonspace = [regex]::Matches($text, '\S').Count
    $other = $nonspace - $dev - $latin
    $total = $dev + $latin + $other
    if ($total -eq 0) { $devPct=0; $latinPct=0; $otherPct=0 }
    else { $devPct=[math]::Round(100*$dev/$total,2); $latinPct=[math]::Round(100*$latin/$total,2); $otherPct=[math]::Round(100*$other/$total,2) }
    $relative = $f.FullName.Substring($root.Length + 1)
    Write-Output "$relative | Devanagari=$devPct% Latin=$latinPct% other=$otherPct% chars=$total"
}

Write-Output '=== TOP EXACT NONBLANK DUPLICATE LINES; PII MASKED ==='
foreach ($name in @('train.txt','tool_train.txt')) {
    $path = Join-Path $root $name
    if (-not (Test-Path -LiteralPath $path)) { Write-Output "$name | missing"; continue }
    Write-Output "FILE $name"
    $groups = Get-Content -LiteralPath $path -Encoding UTF8 | Where-Object { $_.Trim() } | Group-Object | Sort-Object Count -Descending | Select-Object -First 20
    foreach ($group in $groups) {
        $safe = $group.Name
        foreach ($pattern in @($patterns.phone,$patterns.email,$patterns.url)) {
            $safe = [regex]::Replace($safe,$pattern,{ param($m) Mask-Value $m.Value })
        }
        Write-Output "$($group.Count) | $safe"
    }
}

Write-Output '=== COPYRIGHT-RISK STRUCTURE CANDIDATES (heuristic only; no text excerpts) ==='
$copyrightCandidates = [System.Collections.Generic.List[object]]::new()
foreach ($f in ($files | Where-Object { $_.Extension.ToLowerInvariant() -eq '.txt' })) {
    $lines = @(Get-Content -LiteralPath $f.FullName -Encoding UTF8)
    $block = @(); $blockNumber = 0
    foreach ($line in $lines + @('')) {
        if (-not $line.Trim()) {
            if ($block.Count -ge 6) {
                $shortCount = @($block | Where-Object { $_.Length -le 60 }).Count
                $repeatedCount = @($block | Group-Object | Where-Object { $_.Count -gt 1 }).Count
                if ($shortCount -ge [math]::Ceiling($block.Count * 0.75) -or $repeatedCount -gt 0) {
                    $blockNumber++
                    $copyrightCandidates.Add([pscustomobject]@{ File=$f.FullName.Substring($root.Length+1); Block=$blockNumber; Lines=$block.Count; Short=$shortCount; Repeated=$repeatedCount })
                }
            }
            $block = @()
        } else { $block += ,$line }
    }
}
Write-Output "candidate_blocks=$($copyrightCandidates.Count)"
$copyrightCandidates | Select-Object -First 3 | ForEach-Object { Write-Output "$($_.File) | block=$($_.Block) lines=$($_.Lines) short_lines=$($_.Short) repeated_exact_lines=$($_.Repeated)" }
