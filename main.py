import argparse
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

# API KEY FROQ
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

## CHROMA DB STORAGE LOCATION
CHROMA_PATH = os.path.join(os.getcwd(),"chroma")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

PROMPT_TEMPLATE = """
Answer the question based only on the following context:
<context>
{context}
</context>

---

Answer the question based on the above context: {question}
"""


import argparse
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# API KEY GROQ
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

## CHROMA DB STORAGE LOCATION
CHROMA_PATH = os.path.join(os.getcwd(), "chroma")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

PROMPT_TEMPLATE = """
Answer the question based only on the following context:
<context>
{context}
</context>

---

Answer the question based on the above context: {question}
"""


def main():
    print("Loading database ...")
    ## Prepare DB (Loaded safely outside the loop)
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_model,
    )
    print("Database ready. Type 'exit' to quit.\n")

    ## Search DB
    while True:
        # Crucial: This explicitly prints a prompt on your screen so the cursor isn't blank
        query_text = input("Ask Rag: ")

        # Check for break commands
        if query_text.strip().lower() == "exit":
            print("Goodbye!")
            break

        # Skip empty inputs
        if not query_text.strip():
            continue

        results = db.similarity_search_with_relevance_scores(query_text, k=3)

        if not results:
            print("Unable to find matching results (Empty database return)\n")
            continue

        # FIX 1: Access index [0][1] to unpack the float relevance score from the tuple list
        if results[0][1] < 0.7:
            print(
                f"Unable to find matching results (Top score {results[0][1]:.2f} is below 0.7)\n"
            )
            continue

        context_text = "\n\n---\n\n".join(
            [doc.page_content for doc, _score in results]
        )
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(
            context=context_text, question=query_text
        )

        # FIX 2: Confirmed valid OpenAI open-weight model target on Groq
        llm = ChatGroq(model="openai/gpt-oss-120b")
        response_obj = llm.invoke(prompt)

        # FIX 3: Safe object extraction mapping
        response_text = response_obj.content

        ## extract source and create formatted response
        sources = [doc.metadata.get("source", None) for doc, _score in results]
        formatted_response = f"Response: {response_text} \n Sources: {sources}"
        print(formatted_response)
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

