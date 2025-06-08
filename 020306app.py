import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from better_profanity import profanity
import subprocess

st.set_page_config(page_title="Expert Assistant", page_icon="📚")

@st.cache_resource(show_spinner=True)
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    try:
        vectordb = FAISS.load_local("vector_store", embeddings=embeddings, allow_dangerous_deserialization=True)
        return vectordb
    except Exception as e:
        st.error(f"Failed to load vector store: {e}")
        return None

def query_vector_store(vectordb, query, top_k=3):
    return vectordb.similarity_search(query.strip(), k=top_k)

def build_prompt(query, docs):
    context = "\n\n".join([doc.page_content[:1000] for doc in docs])
    return f"""You are an expert assistant. Use the context below to answer the question.

Context:
{context}

Question: {query}

Answer:"""

def query_ollama(prompt, model="phi3"):
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        return result.stdout.strip()
    except Exception as e:
        return f"❌ Ollama error: {e}"

def main():
    st.title("📚 Expert Assistant")

    vectordb = load_vector_store()
    if not vectordb:
        st.stop()

    profanity.load_censor_words()

    query = st.text_area("Ask your question:", height=100)

    if st.button("Ask"):
        if not query.strip():
            st.warning("Please enter a question.")
            return

        if profanity.contains_profanity(query):
            st.error("❌ Please use respectful language when asking questions.")
            return

        with st.spinner("Searching..."):
            results = query_vector_store(vectordb, query)
            if results:
                prompt = build_prompt(query, results)
            else:
                prompt = f"Answer this question as best as you can:\n\n{query}"

            answer = query_ollama(prompt)

        st.markdown("### 💬 Answer:")
        st.write(answer)

if __name__ == "__main__":
    main()
