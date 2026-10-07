import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

st.set_page_config(page_title="RAG Chatbot", page_icon="💬")
st.title("💬 Document Chatbot")

DOC_PATH = "data/document.txt"
INDEX_DIR = "vectorstore"

# API key: Streamlit Secrets (cloud) or environment variable (local)
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = None

if not api_key:
    st.error("GROQ_API_KEY is missing. Add it in Streamlit app Settings → Secrets.")
    st.stop()


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="Building search index (first run only)...")
def get_db():
    embeddings = get_embeddings()
    if os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
        return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    docs = TextLoader(DOC_PATH, encoding="utf-8").load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
    db = FAISS.from_documents(chunks, embeddings)
    os.makedirs(INDEX_DIR, exist_ok=True)
    db.save_local(INDEX_DIR)
    return db


@st.cache_resource
def get_llm():
    # If this model is retired, choose a current one from the Groq console.
    return ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key, temperature=0)


db = get_db()
llm = get_llm()

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

question = st.chat_input("Ask something about the document...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    results = db.similarity_search(question, k=3)
    context = "\n\n".join(r.page_content for r in results)

    prompt = f"""Answer the question using only the context below.
If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = llm.invoke(prompt).content
        st.markdown(answer)
        with st.expander("Retrieved context"):
            st.write(context)

    st.session_state.messages.append({"role": "assistant", "content": answer})
