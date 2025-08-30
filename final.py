import os
import pickle
from pathlib import Path
from dotenv import load_dotenv
from operator import itemgetter


# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# =========================
# 0. Load Environment
# =========================
load_dotenv()
DATA_DIR = Path("data")

# =========================
# 1. Ingest Documents
# =========================
def load_docs():
    docs = []
    if not DATA_DIR.exists():
        print("No data folder found. Create 'data/' and add PDFs or text files.")
        return docs

    for path in DATA_DIR.glob("*"):
        try:
            if path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(path))
                for doc in loader.load():
                    doc.metadata["source"] = path.name
                    docs.append(doc)
            elif path.suffix.lower() in [".txt", ".md"]:
                loader = TextLoader(str(path), encoding="utf-8")
                for doc in loader.load():
                    doc.metadata["source"] = path.name
                    docs.append(doc)
        except Exception as e:
            print(f"Failed to load {path.name}: {e}")
    return docs


def split_docs(docs, chunk_size=800, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(docs)

# =========================
# 2. Create Embeddings + FAISS
# =========================
def build_vectorstore(chunks):
    print(f"Creating embeddings for {len(chunks)} chunks...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(chunks, embedding_model)
    vectordb.save_local("faiss_index")
    return vectordb

# =========================
# 3. Setup Hybrid Retriever
# =========================
def build_hybrid_retriever(vectordb, chunks):
    retriever_vector = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.5}
    )

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 5

    retriever = EnsembleRetriever(
        retrievers=[retriever_vector, bm25_retriever],
        weights=[0.5, 0.5]
    )
    return retriever

# =========================
# 4. Initialize LLM + Prompt
# =========================
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    temperature=0.1,
)
chat_model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate.from_template(
    """
    You are a helpful assistant. Answer the question based only on the following context.
    If you don't know the answer, just say that you don't know. Do not make up an answer.

    Context:
    {context}

    Question:
    {question}
    """
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# =========================
# 5. Build RAG Chain
# =========================
def build_rag_chain(retriever):
    # This sub-chain formats docs → runs prompt → LLM → parses to string
    rag_chain_from_docs = (
        RunnablePassthrough.assign(
            # Keep docs in input for sources, but provide LLM a formatted string
            context=lambda x: format_docs(x["context"])
        )
        | prompt
        | chat_model
        | StrOutputParser()
    )

    # Critically: send ONLY the question string to the retriever
    base = RunnableParallel(
        {
            "context": itemgetter("question") | retriever,  # <-- fix is here
            "question": itemgetter("question"),
        }
    )

    # Attach the answer generated from the formatted docs
    rag_chain_with_source = base.assign(answer=rag_chain_from_docs)
    return rag_chain_with_source

# =========================
# 6. Public Function
# =========================
def get_retriever_and_chain():
    """Main entry point to build retriever + rag_chain for use in Streamlit"""
    print("Loading documents...")
    docs = load_docs()
    if not docs:
        raise ValueError("❌ No documents found in ./data folder")

    print(f"Loaded {len(docs)} documents/pages")
    chunks = split_docs(docs)
    print(f"Split into {len(chunks)} chunks")

    vectordb = build_vectorstore(chunks)
    retriever = build_hybrid_retriever(vectordb, chunks)
    rag_chain_with_source = build_rag_chain(retriever)

    return retriever, rag_chain_with_source

# =========================
# 7. CLI Entry Point
# =========================
if __name__ == "__main__":
    retriever, rag_chain_with_source = get_retriever_and_chain()

    print("\n✅ Ready to answer questions.")
    question = "What is the main topic of the document?"
    print(f"Question: {question}")

    response = rag_chain_with_source.invoke({"question": question})

    print("\n--- Answer ---")
    print(response["answer"])

    print("\n--- Sources ---")
    unique_sources = {doc.metadata.get('source', 'Unknown') for doc in response["context"]}
    for source in unique_sources:
        print(f"- {source}")