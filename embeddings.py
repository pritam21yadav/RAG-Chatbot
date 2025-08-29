import pickle
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

#Loading the chunks we created in ingestion
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

print(f"Loaded {len(chunks)} chunks for embedding")

#Creating embeddings using Hugging Face (sentence-transformers)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#Storing embeddings in FAISS
vectorstore = FAISS.from_documents(chunks, embedding_model)

#Saving the vectorstore for later retrieval
vectorstore.save_local("faiss_index")
print("✅ Embeddings created and FAISS index saved to faiss_index/")