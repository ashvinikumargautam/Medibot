# 🏥 Medibot — AI Medical Chatbot

An end-to-end medical chatbot powered by **Groq (Llama 3)**, **Pinecone** vector search, and **LangChain RAG pipeline**.

---

## 🚀 Tech Stack

- **Flask** — web server
- **LangChain** — RAG pipeline
- **Groq (Llama 3 8B)** — LLM (free, ultra-fast)
- **Pinecone** — vector database
- **HuggingFace sentence-transformers** — embeddings
- **Gunicorn** — production WSGI server

---

## 📁 Project Structure

```
medibot/
├── app.py                  # Flask app (main entry point)
├── store_index.py          # One-time script: loads PDFs → Pinecone
├── requirements.txt        # Pinned dependencies
├── Procfile                # Render/Railway start command
├── runtime.txt             # Python version for deployment
├── render.yaml             # Render configuration
├── setup.py
├── .env                    # Local secrets (never commit!)
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── helper.py           # PDF loader, text splitter, embeddings
│   └── prompt.py           # System prompt
├── Data/                   # Put your medical PDF files here
└── templates/
    └── chat.html           # Chat UI
```

---

## ⚙️ Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/ashvinikumargautam/Medibot.git
cd Medibot

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create .env file with your keys
echo "PINECONE_API_KEY=your_pinecone_key" > .env
echo "GROQ_API_KEY=your_groq_key" >> .env

# 5. Run data ingestion (only once — uploads PDFs to Pinecone)
python store_index.py

# 6. Run the app
python app.py
```

Open **http://localhost:10000** in your browser.

---

## 🌐 Deploy on Render

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your GitHub repo
4. Set these in Render dashboard:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:${PORT:-10000} --workers 1 --timeout 300 --preload`
   - **Environment Variables:** *(see table below)*

> ⚠️ Run `store_index.py` locally **first** to populate your Pinecone index before deploying.

---

## 🔑 Environment Variables

| Variable | Where to get it |
|---|---|
| `PINECONE_API_KEY` | [app.pinecone.io](https://app.pinecone.io) → API Keys |
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) → API Keys |

---

## 🆓 Why Groq instead of Gemini?

| | Gemini Free | Groq Free |
|---|---|---|
| Requests/day | 1,500 | **14,400** |
| Speed | Medium | **Ultra-fast** |
| Model | gemini-1.5-flash | llama3-8b-8192 |
| Cost | Free tier | **Always free** |

---

## ⚠️ Important Notes

- Never commit your `.env` file — it is already in `.gitignore`
- `store_index.py` only needs to be run **once** locally to upload your PDFs to Pinecone
- The `Data/` folder is gitignored — keep your PDFs local only
- Render free tier spins down after 15 minutes of inactivity — first request after sleep takes ~30s