import os
from pathlib import Path
import streamlit as st
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

st.set_page_config(page_title="Pretty Doc Chat", page_icon="🎀", layout="centered")

# ---------- Girly pink theme ----------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Pacifico&family=Quicksand:wght@400;600;700&display=swap');

:root { color-scheme: light; }

.stApp {
    background: linear-gradient(160deg, #fff0f6 0%, #ffe3ef 45%, #f6e6ff 100%);
    font-family: 'Quicksand', sans-serif;
}
.stApp, .stApp p, .stApp li, .stApp label, .stApp span, .stMarkdown {
    color: #5a2a46;
    font-family: 'Quicksand', sans-serif;
}
[data-testid="stHeader"] { background: transparent; }

h1.cute-title {
    font-family: 'Pacifico', cursive;
    color: #e0559a;
    text-align: center;
    font-size: 2.6rem;
    margin-bottom: 0;
    text-shadow: 2px 2px 0 #ffd1e6;
}
p.cute-sub {
    text-align: center;
    color: #b0668f;
    margin-top: 0.2rem;
    margin-bottom: 1.5rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffd9ea 0%, #f9d7f5 100%);
    border-right: 3px dotted #f7a8cf;
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    font-family: 'Pacifico', cursive;
    color: #e0559a;
}

/* Upload box */
[data-testid="stFileUploaderDropzone"] {
    background: #fff7fb;
    border: 2px dashed #f27bb5;
    border-radius: 18px;
}
[data-testid="stFileUploaderDropzone"] * { color: #b0446f !important; }

/* Buttons */
.stButton > button, [data-testid="stFileUploaderDropzone"] button {
    background: linear-gradient(135deg, #ff8fc4, #f26aa8);
    color: white !important;
    border: none;
    border-radius: 999px;
    font-weight: 700;
    padding: 0.4rem 1.2rem;
    box-shadow: 0 4px 12px rgba(242, 106, 168, 0.35);
}
.stButton > button:hover, [data-testid="stFileUploaderDropzone"] button:hover {
    background: linear-gradient(135deg, #f26aa8, #e0559a);
    color: white !important;
}

/* Chat bubbles */
[data-testid="stChatMessage"] {
    background: #fff;
    border: 2px solid #ffc2de;
    border-radius: 22px;
    padding: 0.8rem 1rem;
    box-shadow: 0 4px 14px rgba(240, 130, 180, 0.18);
}

/* Chat input */
[data-testid="stChatInput"] {
    border: 2px solid #f7a8cf;
    border-radius: 999px;
    background: #fff;
}
[data-testid="stChatInput"] textarea { color: #5a2a46 !important; }
[data-testid="stBottom"] > div { background: transparent; }

/* Expanders */
[data-testid="stExpander"] {
    background: #fff7fb;
    border: 1.5px solid #ffc2de;
    border-radius: 14px;
}
</style>
<h1 class="cute-title">🎀 Pretty Doc Chat 🎀</h1>
<p class="cute-sub">Upload a document and ask me anything about it ✨</p>
""",
    unsafe_allow_html=True,
)

# ---------- Settings ----------
BASE = Path(__file__).parent
INDEX_DIR = str(BASE / "vectorstore")
DEFAULT_DOC = None
for candidate in [BASE / "data" / "document.txt", BASE / "document.txt"]:
    if candidate.exists():
        DEFAULT_DOC = str(candidate)
        break


def get_secret(name, default=None):
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets[name]
    except Exception:
        return default


api_key = get_secret("GROQ_API_KEY")
if not api_key:
    st.error("GROQ_API_KEY is missing. Add it in Streamlit app Settings → Secrets.")
    st.stop()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)


@st.cache_resource(show_spinner="Loading embedding model... 💕")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="Reading the default document... 📖")
def get_default_db():
    embeddings = get_embeddings()
    if os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
        return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    if DEFAULT_DOC is None:
        return None
    docs = TextLoader(DEFAULT_DOC, encoding="utf-8").load()
    db = FAISS.from_documents(splitter.split_documents(docs), embeddings)
    os.makedirs(INDEX_DIR, exist_ok=True)
    db.save_local(INDEX_DIR)
    return db


def extract_text(name: str, data: bytes) -> str:
    if name.lower().endswith(".pdf"):
        import io
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    return data.decode("utf-8", errors="ignore")


@st.cache_resource(show_spinner="Reading your document... 📖")
def build_uploaded_db(name: str, data: bytes):
    text = extract_text(name, data).strip()
    if not text:
        return None
    chunks = splitter.split_documents([Document(page_content=text)])
    return FAISS.from_documents(chunks, get_embeddings())


@st.cache_resource
def get_llm():
    # Change the model without editing code: add GROQ_MODEL in Streamlit Secrets.
    model = get_secret("GROQ_MODEL", "openai/gpt-oss-120b")
    return ChatGroq(model=model, api_key=api_key, temperature=0)


# ---------- Sidebar: upload ----------
with st.sidebar:
    st.markdown("## 📎 Your Document")
    uploaded = st.file_uploader(
        "Upload a .txt or .pdf file",
        type=["txt", "pdf"],
        help="Remove the file to go back to the default document.",
    )
    if st.button("🧹 Clear chat"):
        st.session_state.messages = []
        st.rerun()

db = None
doc_label = None
if uploaded is not None:
    try:
        db = build_uploaded_db(uploaded.name, uploaded.getvalue())
    except ImportError:
        st.sidebar.error("PDF support needs `pypdf` in requirements.txt.")
    except Exception as e:
        st.sidebar.error(f"Couldn't read that file: {e}")
    if db is None and uploaded is not None:
        st.sidebar.warning("I couldn't find any text in that file 🥺 (scanned PDFs won't work).")
    else:
        doc_label = uploaded.name
else:
    db = get_default_db()
    doc_label = "default document"

if db is None:
    st.info("Upload a .txt or .pdf file in the sidebar to get started 💖")
    st.stop()

st.sidebar.success(f"Chatting with: {doc_label}")

# Reset chat when the document changes
if st.session_state.get("active_doc") != doc_label:
    st.session_state.active_doc = doc_label
    st.session_state.messages = []

llm = get_llm()

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar="🧑‍💻" if m["role"] == "user" else "🎀"):
        st.markdown(m["content"])

question = st.chat_input("Ask me about your document... 💬")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(question)

    results = db.similarity_search(question, k=3)
    context = "\n\n".join(r.page_content for r in results)

    prompt = f"""Answer the question using only the context below.
If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""

    with st.chat_message("assistant", avatar="🎀"):
        with st.spinner("Thinking... 💭"):
            try:
                answer = llm.invoke(prompt).content
            except Exception as e:
                st.error(f"Groq error: {e}")
                st.stop()
        st.markdown(answer)
        with st.expander("📚 Retrieved context"):
            st.write(context)

    st.session_state.messages.append({"role": "assistant", "content": answer})
