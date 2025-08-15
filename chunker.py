# table_aware_chunker.py

import re
import logging
from typing import List, Dict, Any, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TableAwareChunker:
    """
    A class to perform intelligent, two-level chunking on document analysis results.
    It creates "Parent Chunks" based on Level 1 (#) and Level 2 (##) Markdown headers
    and smaller, table-aware "Child Chunks" for precise vector searching.
    """

    def __init__(self, child_chunk_size: int = 1024):
        self.chunk_size = child_chunk_size

    def process_document(self, analysis_result: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
            
        markdown_content = analysis_result
        parent_chunks = self._create_parent_chunks(markdown_content)
        child_chunks = self._create_table_aware_child_chunks(parent_chunks)
        
        return parent_chunks, child_chunks

    def _create_parent_chunks(self, markdown_content: str) -> List[Dict[str, Any]]:
        """
        Creates large, logical Parent Chunks by splitting the document by
        both Level 1 (#) and Level 2 (##) section headers.
        """
        logging.info("Creating Parent Chunks based on Markdown headers (# and ##)...")
        parent_chunks = []
        doc_id_counter = 0
        
        chunks = re.split(r"\n(?=#{1,2}\s)", markdown_content)

        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            
            parent_chunks.append({
                "parent_id": f"pt_{doc_id_counter}",
                "title": chunk.split('\n')[0].strip(),
                "content": chunk,
            })
            doc_id_counter += 1
        
        logging.info(f"Successfully created {len(parent_chunks)} Parent Chunks.")
        return parent_chunks

    def _create_table_aware_child_chunks(self, parent_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Creates smaller Child Chunks from Parent Chunks, ensuring tables are never split.
        This method requires no changes, as it correctly processes the more granular parent chunks.
        """
        logging.info("Creating table-aware Child Chunks...")
        all_child_chunks = []
        child_id_counter = 0

        for parent in parent_chunks:
            blocks = parent['content'].split('\n\n')
            current_chunk_content = ""

            for block in blocks:
                block = block.strip()
                if not block:
                    continue
                is_table = block.startswith('|') and block.endswith('|')

                if len(current_chunk_content) + len(block) > self.chunk_size and current_chunk_content:
                    all_child_chunks.append(self._create_child_dict(child_id_counter, current_chunk_content, parent))
                    child_id_counter += 1
                    current_chunk_content = ""

                current_chunk_content += block + "\n\n"

                if is_table:
                    all_child_chunks.append(self._create_child_dict(child_id_counter, current_chunk_content, parent))
                    child_id_counter += 1
                    current_chunk_content = ""

            if current_chunk_content.strip():
                all_child_chunks.append(self._create_child_dict(child_id_counter, current_chunk_content, parent))
                child_id_counter += 1
        
        logging.info(f"Successfully created {len(all_child_chunks)} table-aware Child Chunks.")
        return all_child_chunks

    def _create_child_dict(self, child_id: int, content: str, parent_chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Helper to create a formatted child chunk dictionary."""
        return {
            "child_id": f"{parent_chunk['parent_id']}_ch_{child_id}",
            "content": content.strip(),
            "metadata": {
                "parent_id": parent_chunk['parent_id'],
                "parent_title": parent_chunk['title'],
                "full_parent_content": parent_chunk['content']
            }
        }
    

    
MOCK_TABLE_SMALL = "| Header 1 | Header 2 |\n|---|---|\n| Row 1 A | Row 1 B |".strip()

MOCK_MARKDOWN_CONTENT_WITH_SUBSECTIONS = f"""
# Section 1: Introduction

This is the general introduction text, belonging directly to Section 1.

## Sub-section 1.1: Scope

This is the text for the scope sub-section. It should be its own parent chunk.

{MOCK_TABLE_SMALL}

## Sub-section 1.2: Definitions

This is the text for the definitions sub-section, which is another parent chunk.

### heading level-3

This is the type of heading which has 3 hash marks

# Section 2: Coverage Details

This final section has no sub-sections and is its own parent chunk.
"""