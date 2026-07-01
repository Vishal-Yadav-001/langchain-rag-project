from langchain_community.document_loaders import (
    DirectoryLoader,
    UnstructuredMarkdownLoader,
)
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

import os
import shutil
from dotenv import load_dotenv

## Load enviroment variables -> fetch from .env file
load_dotenv()

## setup groq
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

## Paths
DATA_PATH = "data/books"
CHROMA_PATH = os.path.join(os.getcwd(), "chroma")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def main():
    generate_data_store()


def generate_data_store():
    print("\n" + "="*50)
    print("🚀 STARTING RETRIEVAL INGESTION PIPELINE")
    print("="*50)
    
    # Stage 1: Load Data
    print("📂 [1/3] Loading raw text data from source directory...")
    documents = load_documents()
    print(f"✅ Success: Imported {len(documents)} source files.")
    
    # Stage 2: Chunking
    print("\n✂️ [2/3] Initiating hierarchical text chunking (Recursive)...")
    chunks = split_text(documents)
    print(f"✅ Success: Partitioned text into {len(chunks)} overlapping fragments.")
    
    # Stage 3: Embedding & Vector Storage
    print("\n🧠 [3/3] Generating embeddings and syncing with Chroma DB...")
    save_to_chroma(chunks)
    
    print("\n" + "="*50)
    print("🎉 VECTOR DATABASE INDEXING COMPLETED")
    print("="*50 + "\n")


def load_documents():
    loader = DirectoryLoader(
        DATA_PATH, glob="*.md", loader_cls=UnstructuredMarkdownLoader
    )
    document = loader.load()
    return document


def split_text(documents: list[Document]):
    """Convert a big documnet into smaller chunks and have overlap to preserve semantics"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=200, length_function=len, add_start_index=True
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")

    if len(chunks) > 10:
        document = chunks[10]
        print("\n--- Example Chunk Content (Index 10) ---")
        print(document.page_content)
        print(f"Metadata: {document.metadata}\n")

    return chunks



def save_to_chroma(chunks: list[Document]):
    ## clear database first
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    ## Create a new DB from documents
    db = Chroma.from_documents(
        documents=chunks, embedding=embedding_model, persist_directory=CHROMA_PATH
    )
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}")


if __name__ == "__main__":
    main()

