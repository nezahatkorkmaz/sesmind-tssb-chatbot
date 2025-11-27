# SesMind · Stres Farkındalık Chatbotu

SesMind, **TSSB / anksiyete ile ilişkili stres belirtilerini fark etmeye yardımcı olan**, kısa bir tarama sonrası sohbet eden bir destekleyici sohbet asistanıdır.  
Bu proje, **tamamen lokal çalışan** bir LLM mimarisi kullanır; verileriniz **hiçbir şekilde dışarıya gönderilmez**.

---

## Özellikler

- 8 soruluk kısa stres belirti taraması  
  - Yanıt seçenekleri: `Hiç, Nadiren, Bazen, Sık, Çok Sık`
  - Toplam puan üzerinden **düşük / hafif / orta / belirgin** stres kategorisi
- Tarama sonrası:
  - Türkçe, yumuşak ve destekleyici sohbet
  - Nefes egzersizi / grounding gibi basit öneriler
  - Kısa ve anlaşılır yanıtlar
- **Teşhis / tanı koymaz**, ilaç önermez  
- Kullanıcıya **gerekirse bir uzmana başvurması** önerilir

---

## Mimari ve Teknolojiler

- **LLM**: `gemma3:12b` (Ollama üzerinden, *lokal çalışır*)  
- **LLM Orkestrasyonu**: [LangChain](https://www.langchain.com/)  
- **UI**: [Chainlit](https://docs.chainlit.io/)  
- **Model Provider**: [Ollama](https://ollama.com/)

> Bu projede kullanılan LLM, **Ollama ile kendi makinenizde lokal olarak** çalışır.  
> Metinleriniz hiçbir üçüncü parti API'ye gönderilmez.

---

## Kurulum

### 1. Gereksinimler

- Python `3.10+`
- [Ollama](https://ollama.com/) kurulu
- `gemma3:12b` modelinin indirilmiş olması:

```bash
ollama pull gemma3:12b
