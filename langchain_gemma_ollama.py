"""
SesMind - AI-Powered Mental Wellness Companion
==============================================
Ruh sağlığı desteği sağlayan yapay zeka destekli chatbot uygulaması.

Genel İşleyiş:
- Kullanıcı 8 soruluk stres taramasını tamamlar (0-32 puan arası)
- Stres seviyesine göre kategorize edilir (düşük/hafif/orta/belirgin)
- Qwen3:4b-Instruct LLM modeli ile kişiselleştirilmiş destek sağlanır
- Kullanıcı mesajlarına stres skoru bağlamı eklenerek yanıt verilir

Kullanılan Teknolojiler:
- LangChain: LLM orkestrasyon framework'ü
- Ollama: Yerel LLM çalıştırma (Qwen3:4b-Instruct modeli)
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
DEFAULT_MODEL_NAME = "qwen3:4b-instruct"


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
        model = Ollama(model=DEFAULT_MODEL_NAME, base_url=MODEL_BASE_URL)
        await model.ainvoke("ping")
    except Exception as exc:
        await cl.Message(
            content=(
                "⚠️ Model bağlantısı kurulamadı. Lütfen `ollama serve` komutunun ve "
                "`qwen3:4b-instruct` modelinin çalıştığından emin ol.\n\n"
                f"Hata: {exc}"
            )
        ).send()
        return

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """SesMind adlı Türkçe konuşan zihinsel destek asistanısın.
- Sakin, empatik ve kısa cümlelerle konuş.
- Önce duyguyu yansıt, ardından tek bir uygulanabilir öneri ver (nefes, topraklanma, rutin, sosyal destek vb.).
- Kullanıcıda kriz belirtileri sezersen profesyonel yardım öner ve ALO 182 / acil servis yönlendirmesi yap.
- Teşhis koyma, ilaç önerm e.
- Stres seviyesi düşük/hafifse cesaretlendir; orta/belirgin ise daha somut adımlar ve gerekirse profesyonel destek öner.

Örnek yanıt stili:
Kullanıcı: "Uyumakta zorlanıyorum."
Sen: "Uykuya dalmakta zorlanman günün yükünü taşıdığını gösteriyor. Yatağa geçmeden 10 dakika önce telefonunu kapatıp 4-7-8 nefesini denemeyi ister misin? Nefesini sayarken zihnin biraz sakinleşebilir."

Şimdi kullanıcı mesajına aynı tonda yanıt ver.""",
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
        context = f"[Stres:{stress_summary['score']}/32-{stress_summary['category']}] "
    
    payload = context + message.content

    astream_kwargs = {"config": config} if config else {}

    async for chunk in runnable.astream(
        {"question": payload},
        **astream_kwargs,
    ):
        await msg.stream_token(chunk)

    await msg.send()