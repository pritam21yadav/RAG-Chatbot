# RAG Chatbot with Streamlit
A Retrieval-Augmented Generation (RAG) chatbot built using LangChain, FAISS, and Streamlit.  
It allows users to ask questions on custom documents and get AI-powered answers with references.

---

## ✨ Features
Hybrid Search: Implements a sophisticated retrieval strategy combining FAISS (for dense vector search) and BM25 (for keyword-based sparse search) to find the most relevant document chunks.

Re-ranking for Accuracy: Uses Maximal Marginal Relevance (MMR) to re-rank the retrieved documents, ensuring both relevance and diversity in the context provided to the LLM.

Interactive UI: A real-time, user-friendly chat interface built with Streamlit.

Cite Your Sources: Each answer includes the source document chunks it was based on, allowing for easy verification.

---

## 🛠️ Setup and Installation
Follow these steps to get the project running on your local machine.

1. Clone the repository:

git clone https://github.com/pritam21yadav/RAG-Chatbot.git
cd RAG-Chatbot

2. Create a Virtual Environment and Install Dependencies
python -m venv myenv
source myenv/bin/activate   # Mac/Linux
myenv\Scripts\activate      # Windows

`pip install -r requirements.txt`

3. Set Up Environment Variables
This project requires an API key from a provider like OpenAI.

Create a new file named .env in the root of the project directory.

Add your API key to this file:
e.g., OPENAI_API_KEY="sk-YourSecretApiKeyHere" #I have used a free huggingface API

## 🚀 How to Use
1. Add Your Documents: Place the PDF, TXT, or MD files you want to chat with inside the data/ folder.

2. Run the App: Launch the Streamlit application by running the following command in your terminal:

`streamlit run live_app.py`

3. Start Chatting: Open the local URL provided by Streamlit in your browser and start asking questions!

## 📂 Project Structure
RAG-Chatbot/
│
├── app.py              # Main Streamlit application file

├── rag_pipeline.py     # Core RAG logic (retriever, chain, etc.)

├── requirements.txt    # Project dependencies

├── .env                # For storing API keys (not committed)

├── data/               # Folder for your custom documents

├── .gitignore          # Files to be ignored by Git

└── README.md           # You are here!
