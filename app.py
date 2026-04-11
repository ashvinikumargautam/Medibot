from flask import Flask, render_template, request
from src.helper import download_hugging_face_embaddings
from langchain_pinecone import PineconeVectorStore
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
from src.prompt import *

# ----------------------------------------------------
# INITIALIZE FLASK + LOAD ENV
# ----------------------------------------------------
app = Flask(__name__)
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ----------------------------------------------------
# CONFIGURE GOOGLE GEMINI API
# ----------------------------------------------------
genai.configure(api_key=GEMINI_API_KEY)

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


llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",     # ⭐ Correct working model
    temperature=0.2,
    max_output_tokens=600,
    google_api_key=GEMINI_API_KEY,
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
        print("ERROR:", e)
        answer = "⚠️ Server Error: Unable to process your request."

    print("Bot:", answer)
    return str(answer)

# ----------------------------------------------------
# RUN SERVER
# ----------------------------------------------------
if __name__ == "__main__":
    print("🚀 Server running on http://127.0.0.1:8080")
    app.run(host="0.0.0.0", port=8080, debug=True)
