# table_aware_chunker.py

import re
import logging
from typing import List, Dict, Any, Tuple
from bs4 import BeautifulSoup  

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TableAwareChunker:
    """
    A class to perform intelligent chunking, now with the ability to parse
    embedded HTML tables and convert them to clean Markdown.
    """

    def __init__(self, child_chunk_size: int = 1024, max_parent_size: int = 4000):
        self.child_chunk_size = child_chunk_size
        self.max_parent_size = max_parent_size

    async def process_document(self, markdown_content: str) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        initial_parents = await self._create_initial_parent_chunks(markdown_content)
        final_parents = await self._split_oversized_parents(initial_parents)
        parent_content_store = {p['parent_id']: p['content'] for p in final_parents}
        child_chunks = await self._create_table_aware_child_chunks(final_parents)
        print("\n\nCHUNKING COMPLETED")
        return child_chunks, parent_content_store
    
    async def _clean_content(self, text: str) -> str:
        text = re.sub(r'<!-- .*? -->\n?', '', text)
        text = re.sub(r'^\s*#{1,4}\s+', '', text, flags=re.MULTILINE)
        return text.strip()

    async def _create_initial_parent_chunks(self, markdown_content: str) -> List[Dict[str, Any]]:
        parent_chunks = []
        chunks = re.split(r"\n(?=#{1,4}\s)", markdown_content)
        for i, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if not chunk: continue
            parent_chunks.append({"parent_id": f"pt_{i}", "title": chunk.split('\n')[0].strip(), "raw_content": chunk})
        return parent_chunks

    async def _split_oversized_parents(self, parent_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        final_parents = []
        sub_chunk_counter = 0
        for parent in parent_chunks:
            cleaned_content = await self._clean_content(parent['raw_content'])
            if len(cleaned_content.encode('utf-8')) <= self.max_parent_size:
                parent['content'] = cleaned_content
                final_parents.append(parent)
            else:
                logging.warning(f"Parent {parent['parent_id']} is oversized. Splitting...")
                paragraphs = cleaned_content.split('\n\n')
                current_sub_chunk = ""
                for para in paragraphs:
                    if len((current_sub_chunk + para).encode('utf-8')) > self.max_parent_size and current_sub_chunk:
                        sub_parent_id = f"{parent['parent_id']}_sub_{sub_chunk_counter}"
                        final_parents.append({"parent_id": sub_parent_id, "title": f"{parent['title']} (Part {sub_chunk_counter + 1})", "content": current_sub_chunk.strip()})
                        sub_chunk_counter += 1
                        current_sub_chunk = ""
                    current_sub_chunk += para + "\n\n"
                if current_sub_chunk.strip():
                    sub_parent_id = f"{parent['parent_id']}_sub_{sub_chunk_counter}"
                    final_parents.append({"parent_id": sub_parent_id, "title": f"{parent['title']} (Part {sub_chunk_counter + 1})", "content": current_sub_chunk.strip()})
                    sub_chunk_counter += 1
        return final_parents

    async def _create_child_dict(self, child_id: int, content: str, parent_chunk: Dict[str, Any], content_type: str) -> Dict[str, Any]:
        return {"child_id": f"{parent_chunk['parent_id']}_ch_{child_id}", "content": content.strip(), "metadata": {"parent_id": parent_chunk['parent_id'], "parent_title": parent_chunk['title'], "content_type": content_type}}

    
    async def _create_table_aware_child_chunks(self, parent_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Creates smaller Child Chunks, now detecting and converting HTML tables.
        """
        all_child_chunks = []
        child_id_counter = 0

        for parent in parent_chunks:
            blocks = parent['content'].split('\n\n')
            current_chunk_content = ""
            current_chunk_type = 'paragraph'

            for block in blocks:
                block = block.strip()
                if not block: continue
                
                is_html_table = block.startswith('<table>')

                if is_html_table:
                    block = await self._convert_html_table_to_markdown(block)
                    current_chunk_type = 'table'

                if len(current_chunk_content.encode('utf-8')) + len(block.encode('utf-8')) > self.child_chunk_size and current_chunk_content:
                    all_child_chunks.append(await self._create_child_dict(child_id_counter, current_chunk_content, parent, current_chunk_type))
                    child_id_counter += 1
                    current_chunk_content = ""
                    current_chunk_type = 'paragraph'

                current_chunk_content += block + "\n\n"

                if is_html_table:
                    all_child_chunks.append(await self._create_child_dict(child_id_counter, current_chunk_content, parent, current_chunk_type))
                    child_id_counter += 1
                    current_chunk_content = ""
                    current_chunk_type = 'paragraph'

            if current_chunk_content.strip():
                all_child_chunks.append(await self._create_child_dict(child_id_counter, current_chunk_content, parent, current_chunk_type))
                child_id_counter += 1
        
        return all_child_chunks
    
    async def _convert_html_table_to_markdown(self, html_content: str) -> str:
        """
        Parses an HTML table string and converts it into a clean Markdown table.
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            table = soup.find('table')
            if not table:
                return html_content 

            markdown_rows = []
            
            
            for row in table.find_all('tr'):
                cells = [cell.get_text(strip=True) for cell in row.find_all(['td', 'th'])]
                if cells:
                    markdown_rows.append("| " + " | ".join(cells) + " |")

            if len(markdown_rows) > 1 and "| ---" not in markdown_rows[1]:
                header_cell_count = markdown_rows[0].count('|') - 1
                divider = "| " + " | ".join(['---'] * header_cell_count) + " |"
                markdown_rows.insert(1, divider)

            return "\n".join(markdown_rows)
        except Exception as e:
            logging.error(f"Failed to parse HTML table: {e}")
            return html_content
