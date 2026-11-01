# AI-reviewed experiment data contract

Yeh contract sirf is experiment ke liye hai. Yeh human-verified contract `SCHEMA.md` ko replace nahi karta.

Har nonblank JSONL record mein ye fields hon:

- `conversation_id`: split ke waqt ek conversation ko ek hi partition mein rakhne ka stable identifier.
- `user`: user ka prompt.
- `assistant`: AI-drafted assistant reply, jise AI ne is experiment ke liye review kiya.
- `lang`: `hi`, `hinglish`, ya `en`.
- `topic`: `greeting`, `small_talk`, ya `unknown`.
- `source`: hamesha `ai_generated`.
- `reviewed_by_ai`: AI reviewer ka label; yeh human identity ya human approval nahi hai.
- `review_status`: hamesha `ai_reviewed_experiment`.

Validator JSON, required fields, allowed values, non-empty text, unique conversation IDs, exact pair duplicates, prompt overlap, basic script compatibility, aur phone/email/URL patterns check karta hai. Yeh factual correctness, naturalness, ya human approval prove nahi karta.

Training mein sirf is file aur iske matching AI-review manifest ka use hoga. User ke naam par `verified_by` field nahi banegi. Clock aur name-memory replies is experiment ke training records mein nahi honge.
