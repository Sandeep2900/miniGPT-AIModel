# Approved conversation JSONL contract

Har nonblank line ek valid JSON object hoga. Object mein yehi fields honge:

```json
{"user":"kaise ho","assistant":"Main theek hoon, aap kaise hain?","lang":"hinglish","verified_by":"Sandeep Kumar Saket","source":"human_written","topic":"small_talk"}
```

## Fields

- `user`: user ka asli prompt. Isse `User: <text>` ke roop mein model ko diya jayega.
- `assistant`: human-approved target reply. Isse `Assistant: <text>` target banaya jayega.
- `lang`: `hi`, `hinglish`, ya `en`. Yeh pair ki intended language batata hai.
- `verified_by`: review karne wale vyakti ka real naam. `null`, khaali string, ya placeholder qabool nahi honge. Validator string ki sachchai ya reviewer ki identity verify nahi kar sakta; yeh human attestation hai.
- `source`: `human_written` ya `chat_export_anonymized`. Isse provenance record hota hai.
- `topic`: `greeting`, `small_talk`, `name`, `clock`, ya `unknown`. Isse narrow behaviour label hota hai.

## Text aur privacy rules

- `user` aur `assistant` dono non-empty strings hon.
- Phone-like number, email, ya URL pattern wale records reject honge. Regex matches screening flags hain; unka matlab apne aap mein confirmed personal data nahi hota.
- Exact duplicate (`user`, `assistant`) pair reject hoga.
- Script compatibility validator is tarah check karta hai: `hi` ke user aur assistant text mein Devanagari letter hona chahiye; `en` text mein Latin letter hona chahiye aur Devanagari nahi; `hinglish` ke dono text mein Latin letters hone chahiye. Hinglish mein Devanagari bhi aa sakti hai.

`name` aur `clock` topic labels data ko organize karte hain. Ye labels apne aap yeh authorize nahi karte ki model ko naam ya clock ka jawab sikhaya jaye; runtime code ko woh behaviour handle karna hai.

Candidate records approved nahi maane jayenge. Training ke liye file ko is schema ke hisaab se human review karke approve karna hoga.
