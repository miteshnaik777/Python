# Configure your LLM (Groq, OpenAI, Gemini, Ollama, …)

Use **free API keys** or **Ollama (no key)** for answers. No cloud ML platform required.

---

## No API key? Use Ollama (local, free, no sign-up)

**Ollama** runs the LLM on your own machine. No API key, no account, no quota.

1. Install from [ollama.com](https://ollama.com) and open a terminal.
2. Run: `ollama pull llama3.2` (downloads the model once).
3. In `rag_chatbot/.env` set:
   ```env
   LLM_PROVIDER=ollama
   EXTERNAL_LLM_MODEL=llama3.2
   ```
4. Restart the app. Leave all API key fields empty.

Ollama must be running (it usually starts with the app after install). No credits, no rate limits.

---

## Free API keys (cloud LLMs)

| Provider | Free? | Get key | Notes |
|----------|-------|--------|--------|
| **Groq** | Yes, free tier | [console.groq.com](https://console.groq.com) | Fast; good free limits. |
| **Google Gemini** | Yes, free tier | [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) | Sometimes shows "quota 0" in some regions or for new projects; try again later or use Groq/Ollama. |
| **OpenAI (GPT-3.5 / GPT-4)** | **Requires API key** | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) | New accounts get a small free credit, but you **must** create an API key. |

**Why did my Gemini quota show 0?** Free tier can show "limit: 0" for a new project, your region, or after Google’s free-tier changes. Use **Groq** or **Ollama** if you don’t want to wait.

---

## Quick start: OpenAI or Gemini

In `rag_chatbot/.env` set **one** provider and its key:

### OpenAI

1. Get an API key at [platform.openai.com/api-keys](https://platform.openai.com/api-keys).
2. In `.env`:
   ```env
   LLM_PROVIDER=openai
   OPENAI_API_KEY=sk-your-key-here
   ```
3. Optional: `EXTERNAL_LLM_MODEL=gpt-4o-mini` (default) or `gpt-4o`, etc.

### Google Gemini

1. Get an API key at [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey).
2. In `.env`:
   ```env
   LLM_PROVIDER=gemini
   GEMINI_API_KEY=your-key-here
   ```
3. Optional: `EXTERNAL_LLM_MODEL=gemini-2.0-flash` (default). If you get a 404, try `EXTERNAL_LLM_MODEL=gemini-pro`.

Then run: `pip install openai google-generativeai` (if not already), and restart the app.

---

## Other providers

### Groq (free tier)

```env
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_key
EXTERNAL_LLM_MODEL=llama-3.3-70b-versatile
```

### Ollama (local, no key)

1. Install [Ollama](https://ollama.com) and run: `ollama pull llama3.2`
2. In `.env`:

   ```env
   LLM_PROVIDER=ollama
   EXTERNAL_LLM_MODEL=llama3.2
   ```

### Any OpenAI-compatible endpoint (Together, Mistral, etc.)

```env
LLM_PROVIDER=openai_compatible
EXTERNAL_LLM_BASE_URL=https://api.together.xyz/v1
EXTERNAL_LLM_API_KEY=your_key
EXTERNAL_LLM_MODEL=model-name
```

---

## Priority order

- If **`LLM_PROVIDER`** is set (e.g. `openai`, `gemini`, `groq`, `ollama`, `openai_compatible`) → the app uses that **External LLM**.
- Else if **`RAG_USE_DEMO_LLM=true`** → placeholder message (no API call).
- Else → message asking you to set **`LLM_PROVIDER`** and keys (see this file).

---

## `.env` options

| Variable | Meaning | Example |
|----------|--------|--------|
| `LLM_PROVIDER` | Which API to use | `openai`, `gemini`, `groq`, `ollama`, `openai_compatible` |
| `OPENAI_API_KEY` | OpenAI API key | (from platform.openai.com) |
| `GEMINI_API_KEY` | Google Gemini API key | (from aistudio.google.com) |
| `GROQ_API_KEY` | Groq API key | (from console.groq.com) |
| `EXTERNAL_LLM_MODEL` | Model name | `gpt-4o-mini`, `gemini-2.0-flash`, `llama-3.3-70b-versatile` (Groq) |
| `EXTERNAL_LLM_BASE_URL` | For `openai_compatible` only | `https://api.together.xyz/v1` |
| `EXTERNAL_LLM_API_KEY` | Generic key / fallback | For openai_compatible or when no provider-specific key |

Install clients: `pip install openai google-generativeai`
