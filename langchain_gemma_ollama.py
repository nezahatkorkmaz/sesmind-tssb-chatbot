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


# HADS (Hastane Anksiyete ve Depresyon Ölçeği) maddeleri ve puanlama bilgisi
HADS_ITEMS: List[Dict[str, object]] = [
    {
        "id": 1,
        "scale": "A",
        "text": "Kendimi gergin, 'patlayacak gibi' hissediyorum.",
        "options": ["Çoğu zaman", "Birçok zaman", "Zaman zaman, bazen", "Hiçbir zaman"],
        "scores": [3, 2, 1, 0],
    },
    {
        "id": 2,
        "scale": "D",
        "text": "Eskiden zevk aldığım şeylerden hala zevk alıyorum.",
        "options": [
            "Aynı eskisi kadar",
            "Pek eskisi kadar değil",
            "Yalnızca biraz eskisi kadar",
            "Neredeyse hiç eskisi kadar değil",
        ],
        "scores": [0, 1, 2, 3],
    },
    {
        "id": 3,
        "scale": "A",
        "text": "Sanki kötü bir şey olacakmış gibi bir korkuya kapılıyorum.",
        "options": [
            "Kesinlikle öyle ve oldukça da şiddetli",
            "Evet, ama çok da şiddetli değil",
            "Biraz, ama beni endişelendirmiyor.",
            "Hayır, hiç öyle değil",
        ],
        "scores": [3, 2, 1, 0],
    },
    {
        "id": 4,
        "scale": "D",
        "text": "Gülebiliyorum ve olayların komik tarafını görebiliyorum.",
        "options": [
            "Her zaman olduğu kadar",
            "Şimdi pek o kadar değil",
            "Şimdi kesinlikle o kadar değil",
            "Artık hiç değil",
        ],
        "scores": [0, 1, 2, 3],
    },
    {
        "id": 5,
        "scale": "A",
        "text": "Aklımdan endişe verici düşünceler geçiyor.",
        "options": ["Çoğu zaman", "Birçok zaman", "Zaman zaman, ama çok sık değil", "Yalnızca bazen"],
        "scores": [3, 2, 1, 0],
    },
    {
        "id": 6,
        "scale": "D",
        "text": "Kendimi neşeli hissediyorum.",
        "options": ["Hiçbir zaman", "Sık değil", "Bazen", "Çoğu zaman"],
        "scores": [3, 2, 1, 0],
    },
    {
        "id": 7,
        "scale": "A",
        "text": "Rahat rahat oturabiliyorum ve kendimi gevşek hissediyorum.",
        "options": ["Kesinlikle", "Genellikle", "Sık değil", "Hiçbir zaman"],
        "scores": [0, 1, 2, 3],
    },
    {
        "id": 8,
        "scale": "D",
        "text": "Kendimi sanki durgunlaşmış gibi hissediyorum.",
        "options": ["Hemen hemen her zaman", "Çok sık", "Bazen", "Hiçbir zaman"],
        "scores": [3, 2, 1, 0],
    },
    {
        "id": 9,
        "scale": "A",
        "text": "Sanki içim pır pır ediyormuş gibi bir tedirginliğe kapılıyorum.",
        "options": ["Hiçbir zaman", "Bazen", "Oldukça sık", "Çok sık"],
        "scores": [0, 1, 2, 3],
    },
    {
        "id": 10,
        "scale": "D",
        "text": "Dış görünüşüme ilgimi kaybettim.",
        "options": [
            "Kesinlikle",
            "Gerektiği kadar özen göstermiyorum",
            "Pek o kadar özen göstermeyebiliyorum",
            "Her zamanki kadar özen gösteriyorum",
        ],
        "scores": [3, 2, 1, 0],
    },
    {
        "id": 11,
        "scale": "A",
        "text": "Kendimi sanki hep bir şey yapmak zorundaymışım gibi huzursuz hissediyorum.",
        "options": ["Gerçekten de çok fazla", "Oldukça fazla", "Çok fazla değil", "Hiç değil"],
        "scores": [3, 2, 1, 0],
    },
    {
        "id": 12,
        "scale": "D",
        "text": "Olacakları zevkle bekliyorum.",
        "options": [
            "Her zaman olduğu kadar",
            "Her zamankinden biraz daha az",
            "Her zamankinden kesinlikle daha az",
            "Hemen hemen hiç",
        ],
        "scores": [0, 1, 2, 3],
    },
    {
        "id": 13,
        "scale": "A",
        "text": "Aniden panik duygusuna kapılıyorum.",
        "options": ["Gerçekten de çok sık", "Oldukça sık", "Çok sık değil", "Hiçbir zaman"],
        "scores": [3, 2, 1, 0],
    },
    {
        "id": 14,
        "scale": "D",
        "text": "İyi bir kitap, televizyon ya da radyo programından zevk alabiliyorum.",
        "options": ["Sıklıkla", "Bazen", "Pek sık değil", "Çok seyrek"],
        "scores": [0, 1, 2, 3],
    },
]

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


def categorize_hads_subscale(score: int) -> str:
    """0-7 normal, 8-10 sınırda, 11 ve üzeri anormal."""
    if score <= 7:
        return "Normal"
    if score <= 10:
        return "Sınırda"
    return "Anormal"


def _extract_action_value(
    action_response: Union[dict, "cl.Action", None]
) -> Optional[Union[int, str]]:
    if action_response is None:
        return None

    payload = None
    if hasattr(action_response, "payload"):
        payload = getattr(action_response, "payload")
    elif isinstance(action_response, dict):
        payload = action_response.get("payload")

    if isinstance(payload, dict):
        # Tercihen index tabanlı payload kullan (int)
        if "index" in payload:
            return payload.get("index")
        value = payload.get("value")
        if value is not None:
            return value

    if isinstance(action_response, dict):
        return action_response.get("name")

    return None


async def ask_question(index: int) -> int:
    item = HADS_ITEMS[index]
    options: List[str] = item["options"]  # type: ignore[assignment]
    scores: List[int] = item["scores"]  # type: ignore[assignment]

    prefix = QUESTION_PREFIXES[index % len(QUESTION_PREFIXES)]
    content = (
        f"{prefix}\n\n{item['text']}\n\n"
        "Lütfen aşağıdaki seçeneklerden birini işaretle."
    )
    actions = [
        cl.Action(
            name=f"q{index}_{i}",
            label=display,
            payload={"index": i},
        )
        for i, display in enumerate(options)
    ]
    response = await cl.AskActionMessage(content=content, actions=actions).send()
    idx = _extract_action_value(response)

    if not isinstance(idx, int) or idx < 0 or idx >= len(scores):
        # Geçersiz yanıt durumunda aynı soruyu tekrar sor
        return await ask_question(index)

    return scores[idx]


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content=(
            "Merhaba, ben SesMind. Öncelikle Hastane Anksiyete ve Depresyon Ölçeği "
            "(HADÖ) temelli kısa bir tarama yaparak seni daha iyi anlamak istiyorum. "
            "Sorular çoktan seçmeli olacak ve yanıtların bize rehberlik edecek."
        )
    ).send()

    try:
        model = Ollama(model=DEFAULT_MODEL_NAME, base_url=MODEL_BASE_URL)
        await model.ainvoke("ping")
    except Exception as exc:
        await cl.Message(
            content=(
                "⚠️ Model bağlantısı kurulamadı. Lütfen `ollama serve` komutunun ve "
                f"`{DEFAULT_MODEL_NAME}` modelinin çalıştığından emin ol.\n\n"
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
            "Her madde için son birkaç gününü düşünerek sana en yakın gelen seçeneği "
            "işaretlemeni rica edeceğim. Çok düşünmene gerek yok; aklına ilk gelen "
            "yanıt genellikle en doğrusudur."
        )
    ).send()

    answers: List[int] = []
    for idx in range(len(HADS_ITEMS)):
        score = await ask_question(idx)
        answers.append(score)

    anxiety_score = sum(
        s for item, s in zip(HADS_ITEMS, answers) if item["scale"] == "A"  # type: ignore[index]
    )
    depression_score = sum(
        s for item, s in zip(HADS_ITEMS, answers) if item["scale"] == "D"  # type: ignore[index]
    )

    anxiety_cat = categorize_hads_subscale(anxiety_score)
    depression_cat = categorize_hads_subscale(depression_score)

    summary_text = (
        "Yanıtların için teşekkürler.\n\n"
        f"- Anksiyete puanın: {anxiety_score} ({anxiety_cat})\n"
        f"- Depresyon puanın: {depression_score} ({depression_cat})\n\n"
        "Bu ölçek yalnızca tarama amaçlıdır ve tıbbi tanı yerine geçmez. "
        "Belirtilerin seni zorluyorsa bir ruh sağlığı uzmanına başvurmanı öneririm."
    )

    cl.user_session.set(
        "stress_summary",
        {
            "anxiety_score": anxiety_score,
            "anxiety_category": anxiety_cat,
            "depression_score": depression_score,
            "depression_category": depression_cat,
        },
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