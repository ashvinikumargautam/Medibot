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
from src.prompt import *

# ----------------------------------------------------
# INITIALIZE FLASK + LOAD ENV
# ----------------------------------------------------
app = Flask(__name__)
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Set Pinecone key in environment (required by pinecone-client internally)
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY or ""

# ----------------------------------------------------
# LOAD EMBEDDINGS + PINECONE INDEX
# ----------------------------------------------------
embeddings = download_hugging_face_embaddings()
index_name = "medicalbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# ----------------------------------------------------
# GROQ LLM — free tier, very fast
# ----------------------------------------------------
llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0.2,
    max_tokens=600,
    groq_api_key=GROQ_API_KEY,
)

# PROMPT TEMPLATE
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# ----------------------------------------------------
# BUILD RETRIEVAL + QA CHAIN
# ----------------------------------------------------
qa_chain = create_stuff_documents_chain(llm, prompt)
reg_chain = create_retrieval_chain(retriever, qa_chain)

# ----------------------------------------------------
# FLASK ROUTES
# ----------------------------------------------------
@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg")
    print("User:", msg)

    try:
        result = reg_chain.invoke({"input": msg})
        answer = result["answer"]
    except Exception as e:
        traceback.print_exc()
        answer = f"Error details: {str(e)}"

    print("Bot:", answer)
    return str(answer)

# ----------------------------------------------------
# RUN SERVER
# ----------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)