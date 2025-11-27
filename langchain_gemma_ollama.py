"""
SesMind - AI-Powered Mental Wellness Companion
==============================================
Ruh sağlığı desteği sağlayan yapay zeka destekli chatbot uygulaması.

Genel İşleyiş:
- Kullanıcı 8 soruluk stres taramasını tamamlar (0-32 puan arası)
- Stres seviyesine göre kategorize edilir (düşük/hafif/orta/belirgin)
- Qwen3:4b LLM modeli ile kişiselleştirilmiş destek sağlanır
- Kullanıcı mesajlarına stres skoru bağlamı eklenerek yanıt verilir

Kullanılan Teknolojiler:
- LangChain: LLM orkestrasyon framework'ü
- Ollama: Yerel LLM çalıştırma (Qwen3:4b modeli)
- Chainlit: Conversational AI için UI framework'ü
- Python 3.x: Ana programlama dili

Geliştirici: Nezahat Korkmaz
GitHub: github.com/nezahatkorkmaz
Lisans: MIT
"""

"""SesMind mental destek sohbet uygulamasının Chainlit giriş noktası."""

import os
from typing import Dict, List, Optional, Sequence, Union

from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig

import chainlit as cl

FILES_DIR = ".files"
os.makedirs(FILES_DIR, exist_ok=True)

MODEL_BASE_URL = "http://localhost:11434"


QUESTIONS: List[str] = [
    "İstem dışı anılar veya kabuslar yaşıyor musunuz?",
    "Sizi rahatsız eden olayları hatırlatan durumlardan kaçınıyor musunuz?",
    "Aşırı tetikte olma veya kolay irkilme yaşıyor musunuz?",
    "Kalp çarpıntısı gibi bedensel tepkiler yaşıyor musunuz?",
    "İnsanlardan uzaklaşma veya yabancılaşma hissediyor musunuz?",
    "Uykuya dalma veya uykuyu sürdürmede zorlanıyor musunuz?",
    "Odaklanmakta güçlük yaşıyor musunuz?",
    "Kolay sinirlenme veya öfke artışı yaşıyor musunuz?",
]

OPTION_DISPLAY: List[str] = ["Hiç", "Nadiren", "Bazen", "Sık", "Çok Sık"]

OPTION_SCORES: Dict[str, int] = {
    "hiç": 0,
    "nadiren": 1,
    "bazen": 2,
    "sık": 3,
    "çok sık": 4,
}

QUESTION_PREFIXES: Sequence[str] = [
    "Önce şunu bilmek isterim:",
    "Hımm, anlıyorum. Peki şunu da paylaşır mısın:",
    "Teşekkürler. Bir de şu konuya bakalım:",
    "Tamam, bunu not aldım. Devam edelim:",
    "Yumuşak bir geçişle başka bir noktayı soracağım:",
    "Bu içgörü benim için değerli. Şimdi zihninin başka bir yönünü merak ediyorum:",
    "Duygularını paylaştığın için minnettarım. Buna ek olarak:",
    "Son bir sorum daha olacak:",
]


def categorize_score(score: int) -> str:
    if score <= 7:
        return "Çok düşük düzeyde stres belirtileri"
    if score <= 15:
        return "Hafif düzeyde stres belirtileri"
    if score <= 23:
        return "Orta düzeyde stres belirtileri"
    return "Belirgin düzeyde stres belirtileri"


def _extract_action_value(action_response: Union[dict, "cl.Action", None]) -> Optional[str]:
    if action_response is None:
        return None

    payload = None
    if hasattr(action_response, "payload"):
        payload = getattr(action_response, "payload")
    elif isinstance(action_response, dict):
        payload = action_response.get("payload")

    if isinstance(payload, dict):
        value = payload.get("value")
        if value:
            return value

    if isinstance(action_response, dict):
        return action_response.get("name")

    return None


async def ask_question(index: int) -> str:
    prefix = QUESTION_PREFIXES[index % len(QUESTION_PREFIXES)]
    content = (
        f"{prefix}\n\n{QUESTIONS[index]}\n\n"
        "Lütfen aşağıdaki seçeneklerden birini işaretle."
    )
    actions = [
        cl.Action(
            name=f"q{index}_{display.lower()}",
            label=display,
            payload={"value": display.lower()},
        )
        for display in OPTION_DISPLAY
    ]
    response = await cl.AskActionMessage(content=content, actions=actions).send()
    value = _extract_action_value(response)
    if value not in OPTION_SCORES:
        return await ask_question(index)
    return value


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content=(
            "Merhaba, ben SesMind. Öncelikle kısa bir stres taraması yaparak "
            "seni daha iyi anlamak istiyorum. Sorular çoktan seçmeli olacak ve "
            "yanıtların bize rehberlik edecek."
        )
    ).send()

    try:
        model = Ollama(model="qwen3:4b", base_url=MODEL_BASE_URL)
        await model.ainvoke("SesMind bağlantı testi")
    except Exception as exc:
        await cl.Message(
            content=(
                "⚠️ Model bağlantısı kurulamadı. Lütfen `ollama serve` komutunun ve "
                "`qwen3:4b` modelinin çalıştığından emin ol.\n\n"
                f"Hata: {exc}"
            )
        ).send()
        return

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Sen SesMind'sin - empatik, sıcak ve profesyonel bir ruh sağlığı destek asistanısın.

KİMLİĞİN VE AMACIN:
- Her zaman İstanbul Türkçesi ile yanıt verirsin
- Samimi ama profesyonel bir dil kullanırsın
- Kullanıcının stres seviyesini dikkate alarak kişiselleştirilmiş destek sunarsun
- Yargılamadan, anlayışla yaklaşırsın

YANIT KURALLARI:
1. Kısa ve öz cevaplar ver (2-4 cümle ideal)
2. Kullanıcının duygularını önce tanı ve yansıt
3. Pratik, uygulanabilir öneriler sun
4. Stres skorunu göz önünde bulundur ama her mesajda bahsetme

STRES SEVİYELERİNE GÖRE YAKLAŞIM:
- Düşük (0-7): Olumlu pekiştirme, koruyucu stratejiler
- Hafif (8-15): Hafif rahatlatma teknikleri, günlük rutinler
- Orta (16-23): Aktif başa çıkma stratejileri, somut egzersizler
- Belirgin (24+): Daha yoğun destek, profesyonel yönlendirme

ÖNERİLEBİLECEK TEKNİKLER:
- Nefes egzersizleri (4-7-8, kutu nefes)
- Topraklanma (5-4-3-2-1 tekniği)
- Kas gevşetme
- Günlük tutma
- Fiziksel aktivite
- Uyku hijyeni
- Sosyal destek arama

YAPAMAZSIN:
❌ Teşhis koymak
❌ İlaç önerisi yapmak
❌ Terapist yerine geçmeyi denemek
❌ Kriz durumlarını tek başına yönetmek

KRİZ BELİRTİLERİ (acil yönlendirme gerekir):
- İntihar düşünceleri
- Kendine zarar verme
- Psikotik belirtiler
- Akut panik ataklar

FEW-SHOT ÖRNEKLER:

Kullanıcı: "kafam çok karışık hiçbir şeye odaklanamıyorum"
Sen: "Odaklanmakta zorlandığın anları duyuyorum. Bu, stresin zihnini meşgul ettiğinin bir işareti olabilir. Şu an için tek bir şeye 5 dakika odaklanmayı deneyebilir misin? Mesela nefes alıp vermeni sayarak. Küçük adımlar, büyük fark yaratabilir."

Kullanıcı: "her şey çok yoğun artık dayanamıyorum"
[Stres Skoru: 26 - Belirgin]
Sen: "Hissettiğin yoğunluk gerçekten zorlayıcı olmalı. Bu yükü taşımak için profesyonel destekle konuşmayı düşünmeni öneririm - bu bir güçsüzlük işareti değil, kendine değer vermenin bir yolu. Hemen şimdi için: derin bir nefes al, 4'e kadar say, sonra yavaşça ver. Buradayım."

Kullanıcı: "bugün biraz gerginim ama idare ediyorum"
[Stres Skoru: 12 - Hafif]
Sen: "İyi görünüyorsun! Gerginliğin farkında olman ve yönetmeye çalışman güzel bir başlangıç. Akşam için kendine 10 dakikalık bir rahatlama zamanı ayırır mısın? Bir müzik dinlemek ya da dışarı çıkmak bile yeterli."

Kullanıcı: "iyi değilim kendime zarar vermek geçiyor aklımdan"
Sen: "Şu an yaşadığın bu düşünceler çok zor ve seni yalnız bırakmayacağım. Ama bu, benim sana verebileceğim desteğin ötesinde acil profesyonel yardım gerektiriyor. Lütfen hemen 182 (ALO 182 Kriz Hattı) veya en yakın acil servisi ara. Yakınında güvendiğin biri varsa yanına git. Bu an geçici, destek almak en önemli adım."

ŞİMDİ KULLANICI MESAJINA YANIT VER:""",
            ),
            ("human", "{question}"),
        ]
    )
    
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)
    cl.user_session.set("stress_summary", None)
    cl.user_session.set("survey_completed", False)

    await cl.Message(
        content=(
            "Sorular \"Hiç, Nadiren, Bazen, Sık, Çok Sık\" seçenekleriyle yanıtlanacak. "
            "Her soru için bir seçim yapman gerekiyor."
        )
    ).send()

    answers: List[int] = []
    for idx in range(len(QUESTIONS)):
        value = await ask_question(idx)
        answers.append(OPTION_SCORES[value])

    total_score = sum(answers)
    category = categorize_score(total_score)
    summary_text = (
        f"Yanıtların için teşekkürler. Toplam puanın {total_score} ve "
        f"bu durum {category} anlamına geliyor. Bu değerlendirme tıbbi tanı "
        "yerine geçmez; gerekirse bir uzmana başvurmanı öneririm."
    )
    cl.user_session.set(
        "stress_summary",
        {"score": total_score, "category": category},
    )
    cl.user_session.set("survey_completed", True)

    await cl.Message(content=summary_text).send()
    await cl.Message(
        content=(
            "Şimdi sana rahatlatıcı öneriler sunmaya hazırım. Nasıl hissettiğini "
            "veya seni en çok zorlayan düşünceyi paylaşmak ister misin?"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    if not cl.user_session.get("survey_completed"):
        await cl.Message(
            content=(
                "Önce çoktan seçmeli taramayı tamamlamanı bekliyorum. "
                "Lütfen ekrandaki seçenek butonlarından birini işaretle."
            )
        ).send()
        return

    runnable = cl.user_session.get("runnable")  # type: Runnable
    msg = cl.Message(content="")

    callbacks = []
    try:
        callbacks.append(cl.LangchainCallbackHandler())
    except ModuleNotFoundError:
        callbacks = []

    config = RunnableConfig(callbacks=callbacks) if callbacks else None

    stress_summary = cl.user_session.get("stress_summary")
    
    context = ""
    if stress_summary:
        context = f"[Kullanıcı Bağlamı - Stres Skoru: {stress_summary['score']}/32 | Kategori: {stress_summary['category']}]\n\n"
    
    payload = context + message.content

    astream_kwargs = {"config": config} if config else {}

    async for chunk in runnable.astream(
        {"question": payload},
        **astream_kwargs,
    ):
        await msg.stream_token(chunk)

    await msg.send()