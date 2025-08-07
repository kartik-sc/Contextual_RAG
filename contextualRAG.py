from config import Settings
from azure.ai.documentintelligence import DocumentIntelligenceClient
import os
from dotenv import load_dotenv
from typing import List,Dict,Any
import requests
from RAG import DocumentIntelligenceService
import re
import openai
import cohere
import logging
from tqdm import tqdm
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

class ContextualizedRAG:
    def __init__(self, collection_name: str = None):
        
        settings = Settings()
        self.key = settings.llm.api_key
        self.endpoint = settings.llm.base_url

        self.llm_client = openai.OpenAI(
            api_key = self.key,
            base_url = self.endpoint
        )

        self.cohere_client = cohere.ClientV2(
            api_key=settings.cohere.api_key,
            base_url=settings.cohere.base_url,
            client_name=settings.cohere.client_name,
            timeout=settings.cohere.timeout
        )

        self.doc_intel_client = DocumentIntelligenceService()
        
        self.collection = chromadb.CloudClient(
            api_key=settings.chromadb.api_key,
            tenant=settings.chromadb.tenant,
            database=settings.chromadb.database
        )

    def structural_chunking_from_pdf(self, pdf_blob_url) -> List[Dict[str, Any]]:
        """
        Uses Microsoft Document Intelligence to perform section-wise chunking.
        Works if the source is in Markdown Format
        """

        logging.info(f"Analyzing document from URL: {pdf_blob_url}")
        analysis_results = self.doc_intel_client.analyze(source=pdf_blob_url)

        documents = []
        doc_id_counter = 0

        markdown_content = analysis_results["analyzeResult"]["content"]
        chunks = re.split(r"\n(?=#)", markdown_content)

        for chunk in chunks:
            if not chunk.strip():
                continue
            title = chunk.strip().split('\n')[0].strip()

            doc = {
                "doc_id": f"doc_{doc_id_counter}",
                "title": title,
                "chunked_content": chunk.strip(),
            }
            documents.append(doc)
            doc_id_counter += 1

        print("Success")

        return documents, markdown_content
    
    def generate_response(self, prompt1, prompt2, model = "gemini-2.5-flash"):
        """
        Generates a summary for each chunk using the full document as context.
        This is the Anthropic-inspired "Contextual Retrieval" step.
        """

        messages = [
            {
                'role':'user',
                'content':prompt1
            },
            {
                'role':'user',
                'content':prompt2
            }
        ]

        try:
            response = self.llm_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                max_tokens=500,
            )
            return response.choices[0].message.content
        
        except Exception as e:
            logging.error(f"LLM API call failed: {e}")
            return "Error: Could not generate context."
        
    
    def contextualize_documents(self, pdf_blob_url) -> List[Dict[str, Any]]:
        """
        Generates a summary for each chunk using the full document as context.
        """
        
        documents, full_doc_markdown = self.structural_chunking_from_pdf(pdf_blob_url)

        processed_docs = []
        prompt1 = DOCUMENT_CONTEXT_PROMPT.format(doc_content=full_doc_markdown[:20000])
        
        for doc in tqdm(documents, desc="Contextualizing Chunks"):
            prompt2 = CHUNK_CONTEXT_PROMPT.format(chunk_content=doc['chunked_content'])

            summary = self.generate_response(prompt1, prompt2)

            processed_docs.append({
                "doc_id": doc["doc_id"],
                "summary": summary,
                "full_chunk_content": doc["chunked_content"] # This is the payload for final retrieval
            })
        
        return processed_docs
    
    def create_and_store_embeddings(self, processed_docs: List[Dict[str, Any]]):
        """
        Generates embeddings for the summaries using Cohere and stores them in ChromaDB.
        """

        logging.info(f"Generating Vector embeddings for {len(processed_docs)} summaries...")
        
        summaries_to_embed = [doc['summary'] for doc in processed_docs]

        response = self.cohere_client.embed(
            model="embed-v4.0",
            input_type="search_document",
            texts=summaries_to_embed,
            embedding_types=["float"]
        )

        embeddings = response.embeddings.float

        logging.info("Storing summaries, metadata, and embeddings in ChromaDB...")

        self.collection.add(
            ids=[doc["doc_id"] for doc in processed_docs],
            embeddings=embeddings,
            documents=[doc["summary"] for doc in processed_docs], 
            metadatas=[{"full_content": doc["full_chunk_content"]} for doc in processed_docs] 
        )

        logging.info("Storage complete.")

    def data_ingestion(self, pdf_blob_url):
        """
        This function completes the data ingestion into the vector database
        """
        logging.info("DATA ingestion starts")

        try:
            processed_docs = self.contextualize_documents(pdf_blob_url=pdf_blob_url)
            self.create_and_store_embeddings(processed_docs=processed_docs)
            logging.info("Data ingestion Successful")

        except Exception as e:
            print(f"[ERROR]: {e}")
            logging.info(f"[ERROR]: {e}")

    def agentic_retrieval_workflow(self, query: str, top_n: int = 5) -> str:
        """
        Orchestrates the full two-stage RAG process:
        1. Embeds the query.
        2. Performs vector search on summaries.
        3. Reranks the results with Cohere Rerank.
        4. Retrieves the full content for the top results.
        """
        
        query_embedding_response = self.cohere_client.embed(
            model="embed-v4.0",
            input_type="search_document",
            texts=[query],
            embedding_types=["float"]
        )

        query_embedding = query_embedding_response.embeddings.float

        # 2. Vector Search on Summaries
        retrieved_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_n * 3, # Retrieve more results initially for the reranker
            include=["documents", "metadatas"]
        )
        
        # The 'documents' field now contains the summaries
        docs_for_reranking = retrieved_results['documents'][0]

        # 3. Rerank the Summaries for Relevance
        logging.info(f"Reranking the top {len(docs_for_reranking)} summaries.")
        reranked_results = self.cohere_client.rerank(
            model="rerank-v3.5",
            query=query,
            documents=docs_for_reranking,
            top_n=top_n
        )

        final_context_chunks = []
        for result in reranked_results.results:
            # The reranker gives us back the index of the document in the original list
            original_index = result.index
            # Use this index to get the corresponding metadata with the full content
            full_content = retrieved_results['metadatas'][0][original_index]['full_content']
            final_context_chunks.append(full_content)

        logging.info(f"Retrieved {len(final_context_chunks)} final chunks for context.")
        return "\n\n---\n\n".join(final_context_chunks)



pdf_blob_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
response = requests.get(pdf_blob_url)
file_bytes = response.content


questions = [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?",
        "What is the No Claim Discount (NCD) offered in this policy?",
        "Is there a benefit for preventive health check-ups?",
        "How does the policy define a 'Hospital'?",
        "What is the extent of coverage for AYUSH treatments?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?"
]

c = ContextualizedRAG()
c.data_ingestion(pdf_blob_url=pdf_blob_url)
answers = []
c.agentic_retrieval_workflow(query=questions[0])
# for q in questions:
#     answers.append(c.agentic_retrieval_workflow(query=q))