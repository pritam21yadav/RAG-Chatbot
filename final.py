import os
import pickle
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# Loading Environment
load_dotenv()

# Load FAISS Vector DB
print("Loading FAISS DB...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

#Setup Hybrid Retriever
print("Loading retrievers...")

# Vector Retriever with MMR
retriever_vector = vectordb.as_retriever(
    search_type="mmr",   
    search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.5}
)

# BM25 Retriever
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 5

# Hybrid Retriever (Vector + BM25)
retriever = EnsembleRetriever(
    retrievers=[retriever_vector, bm25_retriever],
    weights=[0.5, 0.5]
)
print("Hybrid retriever loaded.")

# Initializing LLM
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    temperature=0.1,
)
chat_model = ChatHuggingFace(llm=llm)

# Prompt Template
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

# RAG Chain with Sources
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Generation part
rag_chain_from_docs = (
    RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
    | prompt
    | chat_model
    | StrOutputParser()
)

# Full pipeline with retriever
rag_chain_with_source = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
).assign(answer=rag_chain_from_docs)

# Ask a Question
if __name__ == "__main__":
    print("\nReady to answer questions.")
    question = "What is the main topic of the document?"
    print(f"Question: {question}")

    response = rag_chain_with_source.invoke(question)

    # Output
    print("\n--- Answer ---")
    print(response["answer"])

    print("\n--- Sources ---")
    unique_sources = {doc.metadata.get('source', 'Unknown') for doc in response["context"]}
    for source in unique_sources:
        print(f"- {source}")