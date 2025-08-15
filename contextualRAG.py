from config import Settings
from chunker import TableAwareChunker
from state import QueryPlan, ReasoningQueryPlan
from extract import DocumentIntelligenceService

from dotenv import load_dotenv
from typing import List,Dict,Any
import requests
import re
import openai
import cohere
import logging
import chromadb

load_dotenv()

DOCUMENT_CONTEXT_PROMPT = """
<document>
{doc_content}
</document>
"""

CHUNK_CONTEXT_PROMPT = """
Here is the chunk we want to situate within the whole document
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context (maximum of 50 words) to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and NOTHING ELSE.
"""

FEW_SHOT_EXAMPLES = [
    {
        "user_query": "I want to reimburse my insurance amount for liver treatment.",
        "reasoning_queries": """- Is this treatment related to alcohol consumption?
- What specific medical procedures were performed?
- Was the treatment received at an in-network or out-of-network hospital?""",
        "search_queries": """- policy exclusions for alcohol-related liver conditions
- coverage details for liver surgery and related treatments
- reimbursement rules for in-network vs out-of-network hospitals
- waiting period for organ-related critical illnesses
- documents required for submitting a major medical claim"""
    },
    {
        "user_query": "I am pregnant and want to know about coverage for delivery.",
        "reasoning_queries": """- How long have you had this insurance policy?
- Is this a routine pregnancy or are there complications?
- Do you plan to use a hospital that is part of the insurance network?""",
        "search_queries": """- maternity and childbirth benefits coverage
- waiting period for pregnancy-related claims
- coverage limits for normal delivery vs. caesarean section
- in-network hospitals and clinics for maternity care
- coverage for newborn baby care and post-natal checkups"""
    },
    {
        "user_query": "I had a bike accident and broke my arm. What do I do?",
        "reasoning_queries": """- Was the treatment performed in an emergency room?
- Does the policy have any exclusions for injuries from hazardous activities or sports?
- Will follow-up care like physiotherapy be required?""",
        "search_queries": """- accidental injury and emergency medical coverage
- policy exclusions related to adventure sports or hazardous activities
- coverage for post-accident rehabilitation and physiotherapy
- procedure for filing an accident insurance claim
- list of documents required for accident reimbursement"""
    }
]


class ContextualizedRAG:
    def __init__(self, collection_name: str = None):
        
        settings = Settings()
        self.key = settings.llm.api_key
        self.endpoint = settings.llm.base_url

        self.llm_client = openai.OpenAI(
            api_key = self.key,
            base_url = self.endpoint
        )

        self.cohere_client = cohere.Client(
            api_key=settings.cohere.api_key,
        )

        self.doc_intel_client = DocumentIntelligenceService()

        # chroma client
        client = chromadb.HttpClient(
        ssl=True,
        host='api.trychroma.com',
        tenant=settings.chromadb.tenant,
        database=settings.chromadb.database,
        headers={
            'x-chroma-token': settings.chromadb.token
        }
        )

        self.collection = client.get_or_create_collection(name="hackrx-sections")

        self.chunker_class = TableAwareChunker(child_chunk_size = 512)

        self.child_chunks = []

    def _extract_content(self, pdf_blob_url):
        """
        Uses Microsoft Document Intelligence to retrieve information in markdown format.
        Chunks the retrieved context into parent chunk and it's child chunk
        """

        analysis_result = self.doc_intel_client.analyze(pdf_blob_url, True)
        markdown_content = analysis_result["analyzeResult"]["content"]

        parent_chunks, child_chunks = self.chunker_class.process_document(markdown_content) 
        self.child_chunks = child_chunks

        return parent_chunks, child_chunks
    
    def _create_and_store_embeddings(self, processed_docs: List[Dict[str, Any]]):
        """
        Generates embeddings for the summaries using Cohere and stores them in ChromaDB.
        """

        print(f"Generating Vector embeddings for {len(processed_docs)} summaries...")
        content_to_embed = [ele['content'] for ele in processed_docs]

        try:
            resp = self.cohere_client.embed(
                model="embed-english-v3.0",
                input_type="search_document",
                texts=content_to_embed,
            )
        except Exception as e:
            resp = self.cohere_client.embed(
                model="embed-multilingual-v3.0",
                input_type="search_document",
                texts=content_to_embed,
            )

        embeddings = resp.embeddings.float_

        self.collection.add(
            ids=[doc["child_id"] for doc in processed_docs],
            embeddings=embeddings,
            documents=[doc["content"] for doc in processed_docs], 
            metadatas=[doc['metadata'] for doc in processed_docs] 
        )

        logging.info("Storage complete.")

    def data_ingestion(self, pdf_blob_url):
        """
        This function completes the data ingestion into the vector database
        """
        logging.info("DATA ingestion starts")

        try:
            _, processed_docs = self._extract_content(pdf_blob_url=pdf_blob_url)
            self._create_and_store_embeddings(processed_docs)

        except Exception as e:
            print(f"[ERROR]: {e}")

    def _query_parser(self, user_query:str):
        """
        Parses the question and generates a few reasoning and clarification
        query.
        """
        
        PLANNER_PROMPT = f"""
        You are an expert insurance claims analyst. Your task is to deconstruct a user's query 
        into a logical plan for investigation.

        Generate "Reasoning Queries". These are clarifying questions a human analyst would 
        need to ask the user to gather all necessary information. They should probe for potential 
        policy exclusions, waiting periods, and network status.
        """
        
        formatted_examples = ""
        for i in range(1,len(FEW_SHOT_EXAMPLES)):
            example = FEW_SHOT_EXAMPLES[i]

            formatted_examples += f"""\n\n
            ==============================
            User Query:
            {example['user_query']}

            Reasoning Queries:
            {example['reasoning_queries']}
            ==============================
            """

        prompt1 = PLANNER_PROMPT + formatted_examples

        try:
            response = self.llm_client.beta.chat.completions.parse(
                model="gemini-2.5-flash",
                messages=[
                    {'role':'system', 'content':prompt1},
                    {'role':'user', 'content': user_query}
                ],
                response_format=ReasoningQueryPlan
            )
            plan = response.choices[0].message.parsed
            return plan
        
        except Exception as e:
            print("Planning failed")

    def _retrieve_and_grade(self, search_queries):
        """
        Executes the retrieval plan, searches for child chunks,
        and returns the content of their unique parents.
        """
        final_results = []

        query_embeddings = self.cohere_client.embed(
            texts=search_queries,
            model="embed-english-v3.0",
            input_type="search_query"
        ).embeddings.float_

        retrieved_results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=12,
            include=["documents","metadatas"]
        )

        try:
            docs_for_reranking = [ans["full_parent_content"] for ans in retrieved_results["metadatas"]]
        except Exception as e:
            print(f"[ERROR]: {e}")
            return final_results
        
        for query in search_queries:
            reranked_results = self.cohere_client.rerank(
                model="rerank-v3.5",
                query=query,
                documents=docs_for_reranking,
                top_n=4
            )

            for result in reranked_results["results"]:
                final_results.append(
                    [
                        result["relevance_score"], 
                        query,
                        docs_for_reranking[result["index"]]
                    ]
                )

        # Sort in descending order
        final_results.sort(key=lambda x:-x[0])

        return final_results

    def _generate_answer(self, final_results:List, user_query:str, reasoning_ques:List[str]):
        """
        Uses the content retrieved from the vector DB and writes the output
        """

        generator_prompt = f"""
            You are a meticulous insurance claims assistant. Your task is to provide a final, 
        comprehensive answer to the user's query based ONLY on the evidence provided 
        from the policy document. Do not invent information or make assumptions.

        Use the 'Reasoning Questions' as a guide to structure your analysis of the evidence. 
        Address each point if possible and cite the information clearly.

        **User's Original Query:**
        <user_query>
        {user_query}
        </user_query>

        **Evidence Corpus from policy document**
        <evidence>
        {final_results[:5]}
        </evidence>

        Provide a clear, structured, and helpful final answer.
        """

        try:
            response = self.llm_client.chat.completions.create(
                model="gemini-2.5-flash", # Better to use a heavy model like 2.5-pro
                messages=[{'role':'user', 'content': generator_prompt}],
                temperature=0.25, 
            )
            return response.choices[0].message.content
        
        except Exception as e:
            print(f"[ERROR]: {e}")
            return "An error occurred while generating the final answer."