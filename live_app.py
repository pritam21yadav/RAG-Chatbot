import streamlit as st
from final import rag_chain_with_source 

st.set_page_config(page_title="RAG Chatbot", layout="wide")

@st.cache_resource(show_spinner=False)
def get_chain():
    return rag_chain_with_source

chain = get_chain()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.sidebar.title("📄 RAG Chatbot")
st.sidebar.markdown("Ask questions on my custom documents.")

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

# Input
user_question = st.chat_input("Type your question here...")

if user_question:
    st.session_state.chat_history.append({"role": "user", "content": user_question})

    response = chain.invoke(user_question)
    answer = response["answer"]
    sources = response["context"]
    unique_sources = sorted({doc.metadata.get("source", "Unknown") for doc in sources})

    st.session_state.chat_history.append(
        {"role": "bot", "content": answer, "sources": unique_sources}
    )

    st.rerun()