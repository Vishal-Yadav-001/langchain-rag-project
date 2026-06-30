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


def main():
    ## Create CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text",type=str,help="Enter your query")
    args = parser.parse_args()
    query_text = args.query_text

    ## Prepare DB
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_model,
    )
    ## Search DB
    results = db.similarity_search_with_relevance_scores(query_text, k=3)

    if not results:
        print(f"Unable to find matching results (Empty database return)")
        return
        
    # Check the score condition on the highest matching entry
    if results[0][1] < 0.7:
        print(f"Unable to find matching results (Top score {results[0][1]:.2f} is below 0.7)")
        return
    
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text,question=query_text)
    print(f"Finalised Prompt {prompt}")

    llm = ChatGroq(model="openai/gpt-oss-120b")
    response_obj = llm.invoke(prompt)
    response_text = response_obj.content

    ## extract source and create formatted response
    sources = [ doc.metadata.get("source",None) for doc,_score in results]
    formatted_response = f"Response: {response_text} \n Sources: {sources}"
    print(formatted_response)



if __name__ == "__main__":
    main()
