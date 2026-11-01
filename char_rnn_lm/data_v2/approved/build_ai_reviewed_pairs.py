# Create a transparent AI-reviewed experiment dataset.

from __future__ import annotations

import json
from pathlib import Path


OUTPUT = Path(__file__).resolve().parent / "ai_reviewed_pairs.jsonl"
REVIEWER = "Codex"

# Each item was drafted for a narrow, non-factual conversation behavior.
# Replies avoid claiming private knowledge or inventing external facts.
PAIRS = [
    # English greetings and small talk.
    ("en", "greeting", "Hey, I have a few quiet minutes.", "Hi! Want to tell me about your day, or keep the chat light?"),
    ("en", "greeting", "Hello again, I felt like checking in.", "Welcome back. What's on your mind this time?"),
    ("en", "greeting", "I could use a little company while I unwind.", "Sure. We can keep it easy; what would you like to talk about?"),
    ("en", "greeting", "Can we just have a casual chat?", "Of course. Pick any small thing that's been on your mind."),
    ("en", "greeting", "Something good happened and I want to tell someone.", "I'm listening. What happened?"),
    ("en", "small_talk", "I've been looking forward to a slower evening.", "That sounds welcome. What would you like to do with the quiet?"),
    ("en", "small_talk", "I feel a bit restless, but nothing is wrong.", "Restless without a clear reason can feel odd. Would talking or a distraction suit you better?"),
    ("en", "small_talk", "I keep replaying an awkward moment.", "Those moments can linger. Do you want to unpack it, or leave it behind for now?"),
    ("en", "small_talk", "I did something I was nervous to try.", "That took effort. How did it feel once you got going?"),
    ("en", "small_talk", "I don't need a solution yet.", "Got it. I'll listen first; tell me what happened in your own way."),
    ("en", "small_talk", "My mood shifted and I can't explain why.", "You don't need a perfect explanation. What do you notice about it right now?"),
    ("en", "small_talk", "I want to untangle one thought.", "Share it as it comes. We can sort through it one piece at a time."),
    ("en", "small_talk", "I have energy to chat, just no topic.", "Let's start small: what was one moment from today that stood out?"),
    ("en", "small_talk", "I'm annoyed with myself for being slow.", "It sounds like you're being hard on yourself. What pace would feel manageable right now?"),
    ("en", "small_talk", "Could you help me switch to something cheerful?", "Sure. Want a playful question, a tiny word game, or a silly imaginary scenario?"),
    ("en", "unknown", "Why did my friend pause before answering?", "I can't know their reason from that detail alone. What else happened in the conversation?"),
    ("en", "unknown", "Can you tell what I meant if I don't explain it?", "Not reliably. Give me a little context and I'll try to understand."),
    ("en", "unknown", "Which choice will definitely make me happy?", "I can't promise an outcome. We can compare what matters to you in each choice."),
    ("en", "unknown", "What was I thinking a moment ago?", "I can't read your thoughts. If you remember a hint, I can help you follow it."),
    ("en", "unknown", "Do you know what I said in another chat?", "I don't have that conversation here. Share the relevant part and I'll continue from there."),

    # Roman Hindi and Hinglish greetings and small talk.
    ("hinglish", "greeting", "Oye, thodi der baat karte hain?", "Haan, bilkul. Aaj kis baare mein baat karne ka mann hai?"),
    ("hinglish", "greeting", "Bas check in karna tha.", "Achha kiya. Abhi tumhara mood kaisa hai?"),
    ("hinglish", "greeting", "Thoda sa company chahiye jab main relax karun.", "Zaroor. Aaram se baat karte hain—kya chal raha hai?"),
    ("hinglish", "greeting", "Aaj koi achhi baat hui, share karun?", "Haan, sunao. Kya hua?"),
    ("hinglish", "greeting", "Kuch casual si baat karni hai.", "Theek hai. Koi chhoti si baat batao jo abhi dimaag mein hai."),
    ("hinglish", "small_talk", "Aaj sab kuch thoda slow lag raha hai.", "Kabhi aisa din bhi hota hai. Tumhe slow pace achha lag raha hai ya bore kar raha hai?"),
    ("hinglish", "small_talk", "Main ek awkward moment ko baar baar yaad kar raha hoon.", "Woh thought atak gaya lagta hai. Uske baare mein baat karni hai ya topic badlein?"),
    ("hinglish", "small_talk", "Mujhe abhi solution nahi chahiye, bas suno.", "Samjha. Pehle tumhari baat sunta hoon; aaram se batao."),
    ("hinglish", "small_talk", "Thoda restless hoon, par wajah samajh nahi aa rahi.", "Wajah clear na ho tab bhi feeling real hoti hai. Abhi tumhe baat karna zyada theek lagega ya distraction?"),
    ("hinglish", "small_talk", "Aaj maine woh kaam shuru kiya jise taal raha tha.", "Shuru karna hi mushkil part hota hai. Start karne mein kis cheez ne help ki?"),
    ("hinglish", "small_talk", "Apne aap se thoda irritate ho raha hoon.", "Kis baat ko lekar khud par pressure daal rahe ho?"),
    ("hinglish", "small_talk", "Mujhe halka sa funny distraction chahiye.", "Chalo ek silly sawaal: agar tumhari chai bol sakti, toh abhi kya kehti?"),
    ("hinglish", "small_talk", "Mere paas baat karne ki energy hai, topic nahi.", "Koi baat nahi. Aaj ka ek chhota sa moment batao jo yaad reh gaya."),
    ("hinglish", "small_talk", "Main apni baat shuru karne mein atak raha hoon.", "Perfect opening ki zaroorat nahi. Jo pehla phrase aaye, wahi se shuru karo."),
    ("hinglish", "small_talk", "Aaj thoda mood swing jaisa lag raha hai.", "Samajh aata hai ki yeh confusing ho sakta hai. Abhi sabse zyada kya feel ho raha hai?"),
    ("hinglish", "unknown", "Mera dost ruk kar kyun bola, tumhe pata hai?", "Sirf itni baat se uski wajah nahi jaan sakta. Us waqt aur kya hua tha?"),
    ("hinglish", "unknown", "Mere dimaag mein kya chal raha hai, guess karo.", "Main thoughts read nahi kar sakta. Ek hint doge toh saath mein samajhne ki koshish karunga."),
    ("hinglish", "unknown", "Kaunsa choice mujhe pakka khush karega?", "Pakka kehna mumkin nahi. Dono choices mein tumhare liye kya important hai, woh dekhte hain."),
    ("hinglish", "unknown", "Pichhli chat mein maine kya bola tha?", "Mere saamne woh chat nahi hai. Tum thoda yaad dila do, main wahin se baat aage badhaunga."),
    ("hinglish", "unknown", "Kya tum bina context ke meri baat samajh jaoge?", "Shayad nahi. Thoda context doge toh main behtar samajh paunga."),

    # Hindi greetings and small talk.
    ("hi", "greeting", "अरे, थोड़ी बातचीत करें?", "ज़रूर। आज किस विषय पर बात करने का मन है?"),
    ("hi", "greeting", "बस हालचाल पूछने आया हूँ।", "अच्छा लगा। अभी आपका मन कैसा है?"),
    ("hi", "greeting", "आराम करते हुए थोड़ी संगत चाहिए।", "ठीक है, आराम से बात करते हैं। अभी क्या चल रहा है?"),
    ("hi", "greeting", "आज एक अच्छी बात हुई, सुनाऊँ?", "हाँ, ज़रूर सुनाइए। क्या हुआ?"),
    ("hi", "greeting", "आज कोई हल्की-सी बातचीत करना चाहता हूँ।", "बिलकुल। मन में जो छोटी-सी बात है, वहीं से शुरू करें।"),
    ("hi", "small_talk", "आज सब कुछ कुछ धीमा लग रहा है।", "धीमा दिन कभी सुकून देता है, कभी उबाऊ लगता है। आपको कैसा लग रहा है?"),
    ("hi", "small_talk", "एक अटपटा पल बार-बार याद आ रहा है।", "ऐसी याद अटक सकती है। क्या आप उसके बारे में बात करना चाहेंगे या विषय बदलें?"),
    ("hi", "small_talk", "अभी मुझे समाधान नहीं चाहिए, बस सुनिए।", "ठीक है। पहले आपकी बात सुनता हूँ—आराम से बताइए।"),
    ("hi", "small_talk", "थोड़ी बेचैनी है, पर वजह समझ नहीं आ रही।", "वजह साफ़ न हो तब भी बेचैनी महसूस हो सकती है। बात करना ठीक लगेगा या ध्यान बँटाना?"),
    ("hi", "small_talk", "आज मैंने टाला हुआ काम शुरू कर दिया।", "शुरुआत करना अक्सर कठिन लगता है। शुरू करने में किस बात ने मदद की?"),
    ("hi", "small_talk", "मैं खुद से थोड़ा नाराज़ हूँ।", "आप खुद पर किस बात का दबाव डाल रहे हैं?"),
    ("hi", "small_talk", "मुझे हल्का-सा मज़ेदार ध्यान भटकाव चाहिए।", "एक कल्पना करें: अगर आपकी चप्पल बोल पाती, तो अभी क्या कहती?"),
    ("hi", "small_talk", "बात करने का मन है, पर कोई विषय नहीं सूझ रहा।", "कोई बात नहीं। आज का एक छोटा-सा पल बताइए जो आपको याद रह गया।"),
    ("hi", "small_talk", "अपनी बात शुरू करने के लिए सही शब्द नहीं मिल रहे।", "सही शुरुआत ज़रूरी नहीं। जो पहला वाक्य मन में आए, वहीं से कहिए।"),
    ("hi", "small_talk", "मेरा मन बदल रहा है और मैं उलझन में हूँ।", "यह उलझन भरा लग सकता है। अभी कौन-सी भावना सबसे स्पष्ट है?"),
    ("hi", "unknown", "मेरा दोस्त जवाब देने से पहले क्यों रुका था?", "सिर्फ़ इस बात से उसकी वजह नहीं जान सकता। उस बातचीत में और क्या हुआ था?"),
    ("hi", "unknown", "बताओ, मेरे मन में अभी क्या चल रहा है?", "मैं आपके विचार नहीं पढ़ सकता। थोड़ा संकेत देंगे तो साथ में समझने की कोशिश करूँगा।"),
    ("hi", "unknown", "कौन-सा फैसला मुझे पक्का खुश करेगा?", "इसकी पक्की गारंटी नहीं दे सकता। दोनों विकल्पों में आपके लिए क्या मायने रखता है, यह देख सकते हैं।"),
    ("hi", "unknown", "पिछली बातचीत में मैंने क्या कहा था?", "वह बातचीत मेरे सामने नहीं है। आप थोड़ा याद दिलाएँ, तो वहीं से आगे बढ़ेंगे।"),
    ("hi", "unknown", "क्या बिना संदर्भ के तुम मेरी बात समझ सकते हो?", "शायद पूरी तरह नहीं। थोड़ा संदर्भ देंगे तो मैं बेहतर समझ पाऊँगा।"),
]


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {OUTPUT}")

    records = []
    for index, (lang, topic, user, assistant) in enumerate(PAIRS, 1):
        records.append(
            {
                "conversation_id": f"ai_exp_{index:03d}",
                "user": user,
                "assistant": assistant,
                "lang": lang,
                "topic": topic,
                "source": "ai_generated",
                "reviewed_by_ai": REVIEWER,
                "review_status": "ai_reviewed_experiment",
            }
        )

    with OUTPUT.open("x", encoding="utf-8", newline="\n") as target:
        for record in records:
            target.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote AI-reviewed experiment records: {OUTPUT}")
    print(f"Records by language: {json.dumps({lang: sum(r['lang'] == lang for r in records) for lang in ('hi', 'hinglish', 'en')}, sort_keys=True)}")
    print(f"Records by topic: {json.dumps({topic: sum(r['topic'] == topic for r in records) for topic in ('greeting', 'small_talk', 'unknown')}, sort_keys=True)}")
    print("Human verified_by field: not used; reviewer is labelled as AI.")


if __name__ == "__main__":
    main()
