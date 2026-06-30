from langchain_community.document_loaders import DirectoryLoader
from langchain_core.vectorstores import Chroma




import os
import shutil
from dotenv import load_dotenv
## Load enviroment variables -> fetch from .env file
load_dotenv()

## setup groq
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

