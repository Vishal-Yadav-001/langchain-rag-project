from langchain_community.document_loaders import DirectoryLoader
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
CHROMA_PATH = os.path.join(os.getcwd() + "chroma")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def main():
    genrate_data_store()

def genrate_data_store():
    documnets = load_documents()
    ## documnets is a big one we nee to make it small to fit into context window, so we chunk
    chunks = split_text(documnets)
    ## after making chunks we will convert it to vectors and store in chromadb
    save_to_chroma(chunks)

def load_documents():
    loader = DirectoryLoader(DATA_PATH,glob="*.md")
    document = loader.load()
    return document

def split_text(documents:list[Document]):
    """ Convert a big documnet into smaller chunks and have overlap to preserve semantics """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 300,
        chunk_overlap = 200,
        length_function = len,
        add_start_index =True
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks

def save_to_chroma(chunks:list[Document]):
    ## clear database first
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    
    ## Create a new DB from documents
    db = Chroma.from_documents(
        chunks = chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_PATH
    )
    db.persist();
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}")


if __name__ == "main":
    main()