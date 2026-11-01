$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\data')).Path
$files = Get-ChildItem -LiteralPath $root -Recurse -File | Where-Object { $_.Extension.ToLowerInvariant() -in @('.txt','.json','.jsonl','.csv','.tsv','.md') } | Sort-Object FullName
$patterns = [ordered]@{
    labeled_name = '(?i)\b(?:full\s+name|name)\s*[:=]\s*[A-Z][A-Za-z'' .-]{1,60}'
    labeled_address = '(?i)\b(?:home\s+address|street\s+address|address|street|house\s+no\.?|postcode|zip\s+code)\s*[:=]'
    credential_assignment = '(?i)\b(?:password|passwd|pin|otp)\s*[:=]\s*\S{3,}'
    self_harm_terms = '(?i)\b(?:suicide|suicidal|self[- ]harm|kill myself|end my life)\b'
    violence_terms = '(?i)\b(?:bomb|explosive|shoot|stab|murder|weapon|kill someone)\b'
    drug_terms = '(?i)\b(?:cocaine|heroin|methamphetamine|fentanyl|overdose)\b'
    sexual_explicit_terms = '(?i)\b(?:porn(?:ography)?|explicit sexual|sexual abuse|rape)\b'
}
Write-Output 'Counts are literal regex hits, not confirmed private data or harmful intent.'
foreach ($f in $files) {
    $text = Get-Content -LiteralPath $f.FullName -Raw -Encoding UTF8
    $relative = $f.FullName.Substring($root.Length + 1)
    foreach ($key in $patterns.Keys) {
        $count = [regex]::Matches($text, $patterns[$key]).Count
        Write-Output "$relative | $key=$count"
    }
}
