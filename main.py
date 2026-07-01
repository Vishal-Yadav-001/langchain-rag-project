import argparse
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from judge import RetrievalJudge

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
    judge = RetrievalJudge(threshold=3)
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

        results = db.similarity_search(query_text, k=4)
        if not results:
            print("Unable to find matching results (Empty database return)\n")
            continue

        ## Quality Check layer
        valid_chunks = judge.evaluate_documents(query_text, results)
        if not valid_chunks:
            print("Fallback: No chunks met the minimum semantic quality gates.\n")
            continue

        context_text = "\n\n---\n\n".join([doc.page_content for doc in valid_chunks])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        # FIX 2: Confirmed valid OpenAI open-weight model target on Groq
        llm = ChatGroq(model="openai/gpt-oss-120b")
        response_obj = llm.invoke(prompt)

        # FIX 3: Safe object extraction mapping
        response_text = response_obj.content

        ## extract source and create formatted response
        sources = [doc.metadata.get("source", "Unknown") for doc in valid_chunks]
        formatted_response = f"Response: {response_text} \nSources: {sources}"
        print(formatted_response)
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
