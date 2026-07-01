# src/judge.py
import json
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
# Standard JSON parsing utility
from langchain_core.output_parsers import SimpleJsonOutputParser


class RetrievalJudge:
    def __init__(self, model_name: str = "llama-3.3-70b-versatile", threshold: int = 3):
        """
        Initializes the LLM Judge with native API JSON Mode validation.
        """
        # CRITICAL: response_format forces Groq to ONLY output a valid JSON object
        self.llm = ChatGroq(
            model=model_name, 
            temperature=0, 
            response_format={"type": "json_object"}
        )
        self.threshold = threshold
        self.output_parser = SimpleJsonOutputParser()

        # The prompt strictly instructs the LLM to format its response as JSON keys
        self.prompt_template = ChatPromptTemplate.from_template("""
        You are a strict QA auditor evaluating how well a retrieved text chunk answers a user question.
        
        Document Chunk:
        <doc>
        {document_content}
        </doc>
        
        User Question: {question}
        
        Analyze the context carefully. You must return your analysis as a valid JSON object with exactly two keys:
        1. "reasoning": A brief string explaining why the text matches or fails the query.
        2. "quality_score": An integer from 0 to 5 based on this rubric:
           - 5: Perfect match (contains the direct answer to the question)
           - 3-4: Partial match (contains useful context, definitions, or background information)
           - 1-2: Poor match (mentions similar words but lacks specific context)
           - 0: Completely irrelevant noise or out-of-bounds topic

        Return ONLY the raw JSON object matching this structure. Do not include markdown formatting like ```json.
        """)
        
        self.chain = self.prompt_template | self.llm | self.output_parser

    def evaluate_documents(self, query: str, candidate_docs: list[Document]) -> list[Document]:
        """
        Grades all candidate chunks and filters out low-quality context blocks using strict JSON mode.
        """
        approved_docs = []
        print("\n--- ⚖️ LLM Judge Quality Audit ---")

        for doc in candidate_docs:
            source_file = doc.metadata.get("source", "Unknown Source")
            start_idx = doc.metadata.get("start_index", "N/A")

            try:
                # Invoke the chain to get a clean Python dictionary back directly
                evaluation = self.chain.invoke({
                    "document_content": doc.page_content,
                    "question": query
                })

                # Safely extract values from the returned dictionary
                reasoning = evaluation.get("reasoning", "No reason provided.")
                score = int(evaluation.get("quality_score", 0))

                print(f"-> Source: {source_file} (Char offset: {start_idx})")
                print(f"   Judge Score: {score}/5")
                print(f"   Judge Reason: {reasoning}")

                if score >= self.threshold:
                    approved_docs.append(doc)
                    print("   Status: ✅ ACCEPTED")
                else:
                    print("   Status: ❌ REJECTED")
                    
            except Exception as e:
                print(f"   ⚠️ Error auditing chunk: {e}. Defaulting to filter exclusion.")
                
            print("-" * 40)
            
        print(f"Audit Complete. Passed {len(approved_docs)}/{len(candidate_docs)} chunks.")
        print(f"Approved Chunks : {approved_docs}")
        print("---------------------------------\n")
        return approved_docs
