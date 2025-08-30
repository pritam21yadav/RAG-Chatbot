import streamlit as st
from final import get_retriever_and_chain

st.set_page_config(page_title="RAG Chatbot", layout="wide")

# =========================
# Load retriever + chain (cached so Streamlit doesn't reload every time)
# =========================
@st.cache_resource(show_spinner=False)
def get_chain():
    with st.spinner("⚙️ Loading documents and building RAG chain..."):
        retriever, rag_chain_with_source = get_retriever_and_chain()
    return retriever, rag_chain_with_source

retriever, chain = get_chain()

# =========================
# Session State for Chat History
# =========================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =========================
# Sidebar
# =========================
st.sidebar.title("📄 RAG Chatbot")
st.sidebar.markdown("Ask questions on your custom documents.")

# =========================
# Main Chat Interface
# =========================
st.title("💬 Document Chatbot")

# Show prior messages
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"🧑 **You:** {msg['content']}")
    else:
        st.markdown(f"🤖 **Bot:** {msg['content']}")
        if msg.get("sources"):
            with st.expander("📂 Sources"):
                for src in msg["sources"]:
                    st.write(f"- {src}")

# =========================
# User Input
# =========================
user_question = st.chat_input("Type your question here...")

if user_question:
    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": user_question})

    # Query the chain with spinner
    with st.spinner("🤖 Thinking..."):
        response = chain.invoke({"question": user_question})

    answer = response["answer"]
    sources = response["context"]
    unique_sources = sorted({doc.metadata.get("source", "Unknown") for doc in sources})

    # Add bot message
    st.session_state.chat_history.append(
        {"role": "bot", "content": answer, "sources": unique_sources}
    )

    # Refresh UI
    st.rerun()