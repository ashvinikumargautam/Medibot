from flask import Flask, render_template, request
from src.helper import download_hugging_face_embaddings

from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv

import os
import traceback


# =========================================================
# APP CONFIGURATION
# =========================================================

app = Flask(__name__)

load_dotenv()


# =========================================================
# ENVIRONMENT VARIABLES
# =========================================================

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Make model configurable from Render
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.3-70b-versatile"
)


# =========================================================
# VALIDATE API KEYS
# =========================================================

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing")


print("========================================")
print("GROQ CONFIGURATION")
print("========================================")

print("GROQ API KEY EXISTS:", bool(GROQ_API_KEY))
print(
    "GROQ KEY PREFIX:",
    GROQ_API_KEY[:8] if GROQ_API_KEY else "MISSING"
)
print("GROQ MODEL:", GROQ_MODEL)

print("========================================")


# =========================================================
# PINECONE CONFIGURATION
# =========================================================

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


# =========================================================
# LOAD EMBEDDINGS
# =========================================================

print("Loading HuggingFace embeddings...")

embeddings = download_hugging_face_embaddings()

print("Embeddings loaded successfully.")


# =========================================================
# CONNECT TO PINECONE
# =========================================================

index_name = "medicalbot"

print("Connecting to Pinecone...")

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)

print("Pinecone connected successfully.")


# =========================================================
# RETRIEVER
# =========================================================

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 3
    }
)


# =========================================================
# GROQ LLM
# =========================================================

print("Initializing Groq LLM...")

llm = ChatGroq(
    model=GROQ_MODEL,
    temperature=0.2,
    max_tokens=600,
    groq_api_key=GROQ_API_KEY,
)

print("Groq LLM initialized.")


# =========================================================
# PROMPT
# =========================================================

from src.prompt import system_prompt


prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])


# =========================================================
# RAG CHAIN
# =========================================================

qa_chain = create_stuff_documents_chain(
    llm,
    prompt
)

reg_chain = create_retrieval_chain(
    retriever,
    qa_chain
)


# =========================================================
# HOME ROUTE
# =========================================================

@app.route("/")
def index():
    return render_template("chat.html")


# =========================================================
# CHAT ROUTE
# =========================================================

@app.route("/get", methods=["POST"])
def chat():

    msg = request.form.get("msg", "").strip()

    print("User:", msg)

    if not msg:
        return "Please enter a question."

    try:

        result = reg_chain.invoke({
            "input": msg
        })

        answer = result["answer"]

    except Exception as e:

        print("========================================")
        print("ERROR")
        print("========================================")

        traceback.print_exc()

        answer = (
            "Sorry, an error occurred while processing your question."
        )

    print("Bot:", answer)

    return str(answer)


# =========================================================
# RUN APP
# =========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            10000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )