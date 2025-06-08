from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
import uvicorn

app = FastAPI()

# Load vector store
print("🔄 Loading vector store...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)

# Connect to Mistral model via Ollama
print("🚀 Connecting to Mistral model...")
llm = Ollama(model="mistral")  # Make sure `ollama run mistral` works

# Set up Retrieval-based QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True
)

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(query: Query):
    print(f"🧠 Received question: {query.question}")
    result = qa_chain(query.question)
    return {
        "question": query.question,
        "answer": result["result"],
        "sources": [doc.metadata for doc in result["source_documents"]]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
