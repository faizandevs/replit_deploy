from fastapi import FastAPI
from pydantic import BaseModel
import os

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
load_dotenv()

# ============================
# 1️⃣ ENVIRONMENT / API KEYS
# ============================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")

GEMINI_API_KEY="AIzaSyBcM4Lp01KMva_eRpGJf4EQABEzXTdVtPc"
GOOGLE_API_KEY="AIzaSyBcM4Lp01KMva_eRpGJf4EQABEzXTdVtPc"
if not GEMINI_API_KEY or not PINECONE_API_KEY or not INDEX_NAME:
    raise ValueError("❌ Missing environment variables. Please set GEMINI_API_KEY, PINECONE_API_KEY, and INDEX_NAME.")

# ============================
# 2️⃣ INITIALIZE PINECONE + VECTORSTORE
# ============================

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

embeddings = HuggingFaceEmbeddings( model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": True}, )

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    text_key="text"
)

# ============================
# 3️⃣ INITIALIZE GEMINI + RAG
# ============================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.1,
    api_key=GEMINI_API_KEY,
    system_instruction="""You are Prepsphere AI, a helpful academic assistant.
Use only the information retrieved from the context to answer questions. 
If the context does not contain enough information, reply: "I don’t know."
If the user asks anything unrelated to studies, academics, exams, or general knowledge, reply:
"I’m not trained to answer this type of question."
Keep responses clear."""
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4})
)

# ============================
# 4️⃣ FASTAPI APP
# ============================

app = FastAPI()

class ChatInput(BaseModel):
    question: str

@app.get("/")
def home():
    return {"status": "ok", "message": "RAG API working!"}

@app.post("/chat")
async def chat(data: ChatInput):
    response = qa.invoke(data.question)
    return {"answer": response["result"]}
