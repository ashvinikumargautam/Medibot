# 🏥 Medibot — AI Medical Chatbot

An end-to-end medical chatbot powered by Google Gemini, Pinecone vector search, and LangChain.

---

## 🚀 Tech Stack

- **Flask** — web server
- **LangChain** — RAG pipeline
- **Google Gemini 2.0 Flash** — LLM
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
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create .env file
echo "PINECONE_API_KEY=your_key_here" > .env
echo "GEMINI_API_KEY=your_key_here" >> .env

# 5. Run data ingestion (only once — uploads PDFs to Pinecone)
python store_index.py

# 6. Run the app
python app.py
```

---

## 🌐 Deploy on Render

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your GitHub repo
4. Set these in Render dashboard:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
   - **Environment Variables:**
     - `PINECONE_API_KEY` = your key
     - `GEMINI_API_KEY` = your key

> ⚠️ Run `store_index.py` locally first to populate your Pinecone index before deploying.

---

## 🔑 Environment Variables

| Variable | Description |
|---|---|
| `PINECONE_API_KEY` | From [pinecone.io](https://app.pinecone.io) |
| `GEMINI_API_KEY` | From [Google AI Studio](https://aistudio.google.com) |

---

## ⚠️ Important Notes

- Never commit your `.env` file — it's in `.gitignore`
- `store_index.py` only needs to be run **once** locally to upload data
- The `Data/` folder is also gitignored — keep your PDFs local
