# For chroma DB embedding function there are two options   
#       1. use the default one
#       2. use cohere embedding function 

# Two step chunking
# summary_embedding: embedding of the concise, situating summary (50 words). This is used for fast approximate 
#                   retrieval from VectorDB.

# content_embedding: embedding of the full chunk (or a compression of it) used for fine-grained reranking 

from utils import trim_bytes, flush_section, is_table, persist_chunk, load_chunk
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
import chromadb.utils.embedding_functions as embedding_functions
import hashlib

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

        self.cohere_client = cohere.Client(
            api_key=settings.cohere.api_key,
        )

        # self.cohere_client = cohere.ClientV2(
        #     api_key=settings.cohere.api_key,
        #     base_url=settings.cohere.base_url,
        #     client_name=settings.cohere.client_name,
        #     timeout=settings.cohere.timeout
        # )

        self.doc_intel_client = DocumentIntelligenceService()
        
        # Local store for full chunks (avoids Chroma size quotas)
        self.store_dir = os.path.join(os.path.dirname(__file__), ".rag_store")
        os.makedirs(self.store_dir, exist_ok=True)

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

    def _markdown_to_chunks(self, markdown_content: str, base_doc_id: str) -> List[Dict[str, Any]]:
        """
        Produce structured chunks
        """

        lines = markdown_content.splitlines()
        sections: List[Dict[str, str]] = []
        current_heading = None
        buffer: List[str] = []

        # Parse top-level sections by markdown headings (#, ##, ###)
        for ln in lines:
            m = re.match(r'^(#{1,6})\s+(.*)$', ln)
            if m:
                # Flush previous section
                sections, buffer = flush_section(sections, current_heading, buffer)
                # Start new section
                current_heading = m.group(2).strip()
            else:
                buffer.append(ln)

        # Flush last section
        sections, buffer = flush_section(sections, current_heading, buffer)

        chunks: List[Dict[str, Any]] = []
        sec_idx = 0
        for sec in sections:
            heading = sec["heading"] or "Untitled Section"
            sec_idx += 1
            parts = [p for p in re.split(r'\n\s*\n', sec.get("text", "")) if p and p.strip()]
            part_idx = 0
            for part in parts:
                txt = part.strip()
                if not txt:
                    continue
                part_idx += 1
                level = "table" if is_table(txt) else "paragraph"
                chunk_id = f"{base_doc_id}#sec{sec_idx}#part{part_idx}"
                chunks.append({
                    "doc_id": base_doc_id,
                    "chunk_id": chunk_id,
                    "title": heading,
                    "section_heading": heading,
                    "chunk_text": txt,
                    "chunk_summary": "",
                    "chunk_level": level,
                })
        return chunks

    def structural_chunking_from_pdf(self, pdf_blob_url) -> List[Dict[str, Any]]:
        """
        Uses Microsoft Document Intelligence to perform section-wise chunking.
        Returns structured chunks according to the required schema.
        """
        logging.info(f"Analyzing document from URL: {pdf_blob_url}")
        analysis_results = self.doc_intel_client.analyze(source=pdf_blob_url)

        markdown_content = analysis_results["analyzeResult"]["content"]

        base_doc_id = f"doc_"

        chunks = self._markdown_to_chunks(markdown_content=markdown_content, base_doc_id=base_doc_id)

        print("Success")
        return chunks, markdown_content
    
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
        prompt1 = DOCUMENT_CONTEXT_PROMPT.format(doc_content=trim_bytes(full_doc_markdown, max_bytes=20000))

        for doc in tqdm(documents, desc="Contextualizing Chunks"):
            prompt2 = CHUNK_CONTEXT_PROMPT.format(chunk_content=doc['chunk_text'])
            summary = self.generate_response(prompt1, prompt2)

            # Persist full chunk to disk and store only a key in Chroma
            store_key = persist_chunk(doc["chunk_text"], self.store_dir)

            processed_docs.append({
                "doc_id": doc["chunk_id"],
                "doc_ref_id": doc["doc_id"],
                "title": doc.get("title", ""),
                "section_heading": doc.get("section_heading", ""),
                "chunk_level": doc.get("chunk_level", "paragraph"),
                "summary": summary,
                "store_key": store_key,
                "full_chunk_content": doc["chunk_text"],
            })
        return processed_docs
    
    def _cohere_embed(self, texts: List[str], input_type: str) -> List[List[float]]:
        """Robust wrapper for Cohere embed across SDK variants."""
        try:
            resp = self.cohere_client.embed(
                model="embed-english-v3.0",
                input_type=input_type,
                texts=texts,
            )
        except Exception as e1:
            logging.warning(f"Cohere embed v3.0 failed: {e1}. Falling back to multilingual.")
            resp = self.cohere_client.embed(
                model="embed-multilingual-v3.0",
                input_type=input_type,
                texts=texts,
            )

        # Handle different SDK return shapes
        if hasattr(resp, "embeddings"):
            emb = getattr(resp.embeddings, "float", None)
            if emb is None:
                emb = resp.embeddings  # already a List[List[float]]
            return emb
        # Fallback (older SDKs might use 'data' etc.)
        if hasattr(resp, "data") and resp.data and hasattr(resp.data[0], "embedding"):
            return [d.embedding for d in resp.data]
        raise RuntimeError("Unexpected Cohere embed response shape")
    
    def create_and_store_embeddings(self, processed_docs: List[Dict[str, Any]]):
        """
        Generates embeddings for the summaries using Cohere and stores them in ChromaDB.
        """

        logging.info(f"Generating Vector embeddings for {len(processed_docs)} summaries...")
        
        summaries_to_embed = [doc['summary'] for doc in processed_docs]

        embeddings = self._cohere_embed(summaries_to_embed, input_type="search_document")

        logging.info("Storing summaries, metadata, and embeddings in ChromaDB...")

        documents_payload = [trim_bytes(doc["summary"], max_bytes=1000) for doc in processed_docs]

        metadatas_payload = [{
            "title": trim_bytes(doc.get("title", ""), max_bytes=256),
            "section_heading": trim_bytes(doc.get("section_heading", ""), max_bytes=256),
            "chunk_level": doc.get("chunk_level", "paragraph"),
            "store_key": doc["store_key"],
            "doc_id": doc.get("doc_ref_id", ""),
        } for doc in processed_docs]

        try:
            self.collection.upsert(
                ids=[doc["doc_id"] for doc in processed_docs],
                embeddings=embeddings,
                documents=documents_payload,
                metadatas=metadatas_payload,
            )
        except AttributeError:
            try:
                self.collection.delete(ids=[doc["doc_id"] for doc in processed_docs])
            except Exception:
                pass
            self.collection.add(
                ids=[doc["doc_id"] for doc in processed_docs],
                embeddings=embeddings,
                documents=documents_payload,
                metadatas=metadatas_payload,
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

    def generate_answer(self, query: str, context: str, model: str = "gemini-2.5-flash") -> str:
        """Return a concise answer strictly based on the provided context."""
        prompt = f"""You are an assistant answering questions about an insurance policy.\n"
            "Answer concisely and only using the provided context. If the answer is not in the context, say: "
            "\"I couldn't find that in the policy.\"\n\n"
            f"Context:
                  <content>
                    {trim_bytes(context, max_bytes=12000)}
                  </content>\n\n"
            f"Question:
                  <query>
                        {query}
                  </query>\n\n"
            Answer:
            
            Generate only the related to the query and nothing else.
            """
        try:
            resp = self.llm_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Answer generation failed: {e}")
            return "I couldn't generate an answer at this time."
    
    def agentic_retrieval_workflow(self, query: str, top_n: int = 5) -> str:
        """
        Orchestrates the full two-stage RAG process:
        1. Embeds the query.
        2. Performs vector search on summaries.
        3. Reranks the results with Cohere Rerank.
        4. Retrieves the full content for the top results.
        """
        
        q_emb = self._cohere_embed([query], input_type="search_query")[0]

        retrieved_results = self.collection.query(
            query_embeddings=[q_emb],
            n_results=top_n * 3,
            include=["documents", "metadatas"]
        )

        docs_for_reranking = retrieved_results.get('documents', [[]])[0]
        metas = retrieved_results.get('metadatas', [[]])[0]

        if not docs_for_reranking:
            return "No relevant context found."

        logging.info(f"Reranking the top {len(docs_for_reranking)} summaries.")
        try:
            reranked_results = self.cohere_client.rerank(
                model="rerank-english-v3.0",
                query=query,
                documents=docs_for_reranking,
                top_n=min(top_n, len(docs_for_reranking))
            )
            top_indices = [r.index for r in reranked_results.results]
        except Exception as e:
            logging.warning(f"Rerank failed: {e}. Using initial order.")
            top_indices = list(range(min(top_n, len(docs_for_reranking))))

        final_context_chunks = []
        for idx in top_indices:
            store_key = metas[idx].get("store_key")
            if store_key:
                full_content = load_chunk(store_key, self.store_dir)
            else:
                full_content = metas[idx].get("full_content", "")
            final_context_chunks.append(full_content)

        logging.info(f"Retrieved {len(final_context_chunks)} final chunks for context.")
        
        context = "\n\n---\n\n".join(final_context_chunks)
        return self.generate_answer(query=query, context=context)

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
ans = c.agentic_retrieval_workflow(query=questions[0])


for q in questions:
    answers.append(c.agentic_retrieval_workflow(query=q))