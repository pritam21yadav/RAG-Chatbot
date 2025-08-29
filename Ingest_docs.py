from pathlib import Path
import pickle
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

#Folder containing documents
DATA_DIR = Path("data") 

#Load all documents
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

#Split documents into chunks
def split_docs(docs, chunk_size=800, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(docs)

#Main execution
def main():
    docs = load_docs()
    if not docs:
        print("No documents found in ./data")
        exit()

    print(f"Loaded {len(docs)} documents/pages")
    chunks = split_docs(docs)
    print(f"Split into {len(chunks)} chunks")

    #Saving chunks to disk for later steps
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
        print("Chunks saved to chunks.pkl")

    return chunks

if __name__ == "__main__":
    chunks = main()