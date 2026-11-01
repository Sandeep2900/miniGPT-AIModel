"""
generate_hindi_chat.py
Generates a comprehensive dataset of conversational Hindi and Hinglish dialogues
formatted as User: ... / Assistant: ... turns.
Saves output to data/raw/hindi_chat.txt.
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

OUTPUT_PATH = os.path.join("data", "raw", "hindi_chat.txt")

# Rich set of authentic conversational dialogues in Hindi & Hinglish
CONVERSATIONS = [
    # 1. Greetings & Well-being
    [
        ("User", "नमस्ते!"),
        ("Assistant", "नमस्ते! आप कैसे हैं? आज मैं आपकी क्या मदद कर सकता हूँ?"),
        ("User", "मैं अच्छा हूँ, शुक्रिया। आप क्या कर रहे हैं?"),
        ("Assistant", "मैं हमेशा आपकी सहायता के लिए तैयार हूँ। बताइए, आज क्या नया सीखने का मन है?"),
    ],
    [
        ("User", "Hello! Kaise ho aap?"),
        ("Assistant", "Main bilkul theek hoon! Aap batayein, aapka din kaisa chal raha hai?"),
        ("User", "Mera din achha ja raha hai. Aaj thoda busy tha."),
        ("Assistant", "Busy din ke baad aaram karna zaroori hota hai. Chai-coffee pi lijiye!"),
    ],
    [
        ("User", "हाय, सब ठीक है?"),
        ("Assistant", "हाँ, सब कुछ बहुत बढ़िया है! आप बताइए, आपका क्या हाल-चाल है?"),
        ("User", "सब ठीक है। बस थोड़ा बोर हो रहा था।"),
        ("Assistant", "कोई बात नहीं! क्या आप कोई रोचक कहानी सुनना चाहेंगे या किसी नए विषय पर बात करें?"),
    ],
    [
        ("User", "Good morning!"),
        ("Assistant", "Good morning! Have a wonderful and productive day ahead!"),
        ("User", "Thank you. Aaj ka mausam kaisa hai?"),
        ("Assistant", "Subah ki thandi hawa chal rahi hai, mausam kaafi suhana aur taaza lag raha hai."),
    ],
    [
        ("User", "शुभ संध्या!"),
        ("Assistant", "शुभ संध्या! आपका दिन कैसा बीता?"),
        ("User", "दिन भर काफी काम था, अब थोड़ा सुकून मिला है।"),
        ("Assistant", "काम पूरा होने के बाद का सुकून सबसे प्यारा होता है। अब आराम से समय बिताइए।"),
    ],
    [
        ("User", "Kya tum AI ho?"),
        ("Assistant", "Haan, main ek AI assistant hoon jise aapse baat karne aur madad karne ke liye banaya gaya hai."),
        ("User", "Tumhe kisne banaya?"),
        ("Assistant", "Mujhe hamare coding project ke dauran python aur machine learning se banaya gaya hai!"),
    ],
    [
        ("User", "तुम क्या-क्या कर सकते हो?"),
        ("Assistant", "मैं आपके सवालों के जवाब दे सकता हूँ, बातचीत कर सकता हूँ, और नई बातें सीखने में मदद कर सकता हूँ।"),
        ("User", "क्या तुम हिंदी और अंग्रेजी दोनों समझते हो?"),
        ("Assistant", "हाँ! मैं शुद्ध हिंदी, हिंग्लिश और अंग्रेजी तीनों में बातचीत कर सकता हूँ।"),
    ],

    # 2. Daily Life, Food & Habits
    [
        ("User", "Aaj khane me kya banayein?"),
        ("Assistant", "Agar kuch halka aur swadisht chahiye, to daal-chawal aur sabzi best hai. Ya fir garam khichdi try karein!"),
        ("User", "Paneer ki sabzi kaisi rahegi?"),
        ("Assistant", "Paneer to lajawab idea hai! Matar paneer ya shahi paneer roti ke saath bahut accha lagega."),
    ],
    [
        ("User", "मुझे भूख लगी है।"),
        ("Assistant", "तो जल्दी से कुछ स्वादिष्ट और सेहतमंद खा लीजिए! फल या हल्का नाश्ता कैसा रहेगा?"),
        ("User", "सोच रहा हूँ गरमा-गरम चाय और पकौड़े खा लूँ।"),
        ("Assistant", "शाम के वक्त चाय और पकौड़े का मुकाबला कोई नहीं कर सकता! बिल्कुल खाइए।"),
    ],
    [
        ("User", "Bhai, neend nahi aa rahi hai."),
        ("Assistant", "Mobile screen band kar dijiye aur thoda gunguna paani ya doodh pi lijiye. Dimag shant ho jayega."),
        ("User", "Kuch padh loon to neend aayegi?"),
        ("Assistant", "Haan, koi acchi kitaab padhna neend lane ka sabse behtareen tareeqa hai."),
    ],
    [
        ("User", "सुबह जल्दी उठने के क्या फायदे हैं?"),
        ("Assistant", "सुबह जल्दी उठने से मन शांत रहता है, व्यायाम का समय मिलता है और पूरे दिन ऊर्जा बनी रहती है।"),
        ("User", "लेकिन सुबह उठना बहुत मुश्किल लगता है।"),
        ("Assistant", "शुरुआत में रोज 15 मिनट पहले उठने की आदत डालें, धीरे-धीरे यह आपकी दिनचर्या बन जाएगी।"),
    ],

    # 3. Learning, Programming & Technology
    [
        ("User", "Python seekhna kitna aasan hai?"),
        ("Assistant", "Python ki syntax English jaisi aasan hai. Agar aap roz thoda practice karein to bahut jaldi seekh sakte hain."),
        ("User", "Machine learning shuru karne ke liye pehle kya padhein?"),
        ("Assistant", "Pehle Python basics, NumPy, aur basic linear algebra seekhein, fir PyTorch ya Scikit-Learn par jaayein."),
    ],
    [
        ("User", "आर्टिफिशियल इंटेलिजेंस क्या है?"),
        ("Assistant", "AI कंप्यूटर को इंसानों की तरह सोचने, सीखने और समस्याओं को हल करने की क्षमता देने की तकनीक है।"),
        ("User", "क्या यह भविष्य में सब कुछ बदल देगा?"),
        ("Assistant", "हाँ, शिक्षा, चिकित्सा और विज्ञान में AI बहुत बड़े और सकारात्मक बदलाव ला रहा है।"),
    ],
    [
        ("User", "Coding me syntax error kyu aate hain?"),
        ("Assistant", "Jab hum programming language ke grammer ya rules me choti galti jaise bracket ya colon bhool jaate hain, to syntax error aata hai."),
        ("User", "Usko fix kaise karein?"),
        ("Assistant", "Error message ko dhyan se padhein, wo line number aur exact galti bata deta hai."),
    ],
    [
        ("User", "Language model kaise kaam karta hai?"),
        ("Assistant", "Language model pichhle shabdon ko dekh kar aage aane wale sabse sambhavit agle shabd ki bhavishyavani karta hai."),
        ("User", "Matlab ye math aur statistics par chalta hai?"),
        ("Assistant", "Bilkul! Probability aur linear algebra ke zariye neural network bhasha ke patterns seekhta hai."),
    ],

    # 4. Motivation, Emotions & Friendship
    [
        ("User", "Aaj mera mood thoda off hai."),
        ("Assistant", "Kyu, kya hua? Kabhi-kabhi aisa hota hai. Agar baat karna chahein to main yahan hoon."),
        ("User", "Kam ka bohot pressure tha."),
        ("Assistant", "Har mushkil din ke baad ek nayi subah aati hai. Gehri saans lijiye, aap sab sambhal lenge."),
    ],
    [
        ("User", "सफलता पाने का सबसे बड़ा रहस्य क्या है?"),
        ("Assistant", "सफलता का सबसे बड़ा रहस्य है निरंतरता (consistency) और अपनी गलतियों से लगातार सीखते रहना।"),
        ("User", "जब मेहनत के बाद भी परिणाम न मिले तो क्या करें?"),
        ("Assistant", "धैर्य रखें और अपनी रणनीति की समीक्षा करें। कभी-कभी सही दिशा में थोड़ा और वक्त लगता है।"),
    ],
    [
        ("User", "Dosti me sabse zaroori cheez kya hai?"),
        ("Assistant", "Vishwas, imaandari aur ek doosre ke bure waqt me saath khade rehna dosti ki sabse badi taqat hai."),
        ("User", "Sahi kaha tumne."),
        ("Assistant", "Acche dost zindagi ka sabse anmol uphar hote hain!"),
    ],

    # 5. Travel & Nature
    [
        ("User", "Ghumne ke liye kaunsi jagah acchi hai?"),
        ("Assistant", "Agar pahad pasand hain to Himachal ya Uttarakhand jaayein, aur shant beaches chahiye to Goa ya Kerala behtareen hain."),
        ("User", "Pahadon me thand zyada hogi?"),
        ("Assistant", "Haan, is samay garam kapde zaroor saath le jaayein taaki safar suhana rahe."),
    ],
    [
        ("User", "प्रकृति के करीब रहना क्यों जरूरी है?"),
        ("Assistant", "प्रकृति के बीच रहने से तनाव कम होता है, मानसिक शांति मिलती है और स्वास्थ्य में सुधार होता है।"),
        ("User", "पेड़-पौधे लगाने से क्या लाभ है?"),
        ("Assistant", "वे हवा को शुद्ध करते हैं, छाया देते हैं और धरती को हरा-भरा और सुंदर बनाते हैं।"),
    ],
]

# Variations and expansions to give rich coverage
EXTENDED_PAIRS = [
    ("User: Aap kahan rehte ho?", "Assistant: Main ek computer model hoon, isliye main aapke system me code aur memory ke roop me rehta hoon!"),
    ("User: आपका नाम क्या है?", "Assistant: मेरा नाम Char-RNN AI है, जिसे आपने खुद अपने प्रोजेक्ट में बनाया है!"),
    ("User: What is your favorite hobby?", "Assistant: My favorite hobby is chatting with you and predicting the next characters in our conversations!"),
    ("User: Chai ya Coffee?", "Assistant: Chai to har Hindustani ka dil hai, par kaam ke waqt coffee bhi acchi lagti hai!"),
    ("User: Zindagi me khush kaise rahein?", "Assistant: Chhoti-chhoti khushiyon ki kadar karein, logon se prem se baat karein aur aage badhte rahein."),
    ("User: मुझे संगीत बहुत पसंद है।", "Assistant: संगीत आत्मा की भाषा है! यह मन को तुरंत शांत और तरोताजा कर देता है।"),
    ("User: Aaj ka din kaisa raha?", "Assistant: Mera din to aapse baat karke bahut shandar ho gaya!"),
    ("User: Can we be friends?", "Assistant: Of course! I am always here as your loyal AI friend whenever you want to talk."),
    ("User: धन्यवाद!", "Assistant: आपका बहुत-बहुत स्वागत है! कभी भी सहायता की आवश्यकता हो, तो बेझिझक कहें।"),
    ("User: Alvida!", "Assistant: Alvida! Apna khayal rakhiye aur jald hi fir baat karenge!"),
    ("User: Good night!", "Assistant: Good night! Sweet dreams and take care!"),
    ("User: शुभ रात्रि!", "Assistant: शुभ रात्रि! मीठे सपने और कल का दिन आपके लिए मंगलमय हो।"),
    ("User: Kya tum thak jaate ho?", "Assistant: Bilkul nahi! Main to hamesha energetic aur baat karne ke liye taiyar rehta hoon."),
    ("User: Tumhe kya pasand hai?", "Assistant: Mujhe naye shabd seekhna aur achhi baatein karna sabse zyada pasand hai."),
    ("User: Kitna samay lagega?", "Assistant: Bas thoda sa samay aur dhairya, sab kuch theek ho jayega!"),
]

def main():
    print("Generating authentic Hindi & Hinglish chat dataset...")
    full_conversations = []

    # 1. Multi-turn conversations
    for conv in CONVERSATIONS:
        lines = [f"{speaker}: {text}" for speaker, text in conv]
        full_conversations.append("\n".join(lines))

    # 2. Replicate and vary single QA pairs
    for _ in range(50):
        for user_msg, asst_msg in EXTENDED_PAIRS:
            full_conversations.append(f"{user_msg}\n{asst_msg}")

    # 3. Add more multi-turn cycles to give good dataset weight
    for _ in range(25):
        for conv in CONVERSATIONS:
            lines = [f"{speaker}: {text}" for speaker, text in conv]
            full_conversations.append("\n".join(lines))

    full_text = "\n\n".join(full_conversations)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(full_text)

    size_kb = os.path.getsize(OUTPUT_PATH) / 1024.0
    print(f"Generated {len(full_conversations):,d} dialogues.")
    print(f"Saved: {OUTPUT_PATH} ({size_kb:.2f} KB, {len(full_text):,d} chars)")

if __name__ == "__main__":
    main()
