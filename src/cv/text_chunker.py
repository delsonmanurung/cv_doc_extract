import re
import sys
from typing import List, Dict, Any
from config.models import ParsedBlock
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=30, fill='█'):
    percent = ('{0:.' + str(decimals) + 'f}').format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    if iteration == total:
        print()

class TextChunker:

    def __init__(self, max_chunk_size: int = 1500, chunk_overlap: int = 150):
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap

    def blocks_to_markdown(self, blocks: List[ParsedBlock]) -> str:
        markdown_lines = []
        total_blocks = len(blocks)
        for idx, block in enumerate(blocks):
            print_progress_bar(idx + 1, total_blocks, prefix='Formatting blocks...  ', suffix=f'Block {idx + 1}/{total_blocks}', length=30)
            if block.block_type == 'Title' and block.content:
                title = block.content.strip()
                if re.match(r'^\d+\.\d+\.\d+\s+', title): header = f'#### {title}'
                elif re.match(r'^\d+\.\d+\s+', title): header = f'### {title}'
                elif re.match(r'^\d+\s+', title): header = f'## {title}'
                else: header = f'# {title}'
                markdown_lines.append(f'\n{header}\n')
            elif block.block_type == 'Text' and block.content:
                page_marker = f'[PAGE_NUM:{block.page_number}]'
                markdown_lines.append(f'\n{page_marker}\n{block.content}\n')
        return ''.join(markdown_lines)

    def chunk_document(self, blocks: List[ParsedBlock]) -> List[Dict[str, Any]]:
        markdown_text = self.blocks_to_markdown(blocks)
        headers_to_split_on = [('#', 'Header 1'), ('##', 'Header 2'), ('###', 'Header 3'), ('####', 'Header 4')]
        
        print_progress_bar(1, 3, prefix='Splitting text...     ', suffix='Running header splitter', length=30)
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        docs = splitter.split_text(markdown_text)
        
        print_progress_bar(2, 3, prefix='Splitting text...     ', suffix='Enforcing token limits', length=30)
        char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = []
        total_docs = len(docs)
        for idx, doc in enumerate(docs):
            print_progress_bar(idx + 1, total_docs, prefix='Cleaning metadata...  ', suffix=f'Chunk {idx + 1}/{total_docs}', length=30)
            text = doc.page_content
            pages_found = [int(p) for p in re.findall(r'\[PAGE_NUM:(\d+)\]', text)]
            cleaned_text = re.sub(r'\[PAGE_NUM:\d+\]', '', text).strip()
            
            if not cleaned_text: continue
                
            pages = sorted(list(set(pages_found)))
            metadata = dict(doc.metadata)
            metadata['pages'] = pages
            
            sub_chunks = char_splitter.split_text(cleaned_text)
            
            for sub_chunk in sub_chunks:
                if sub_chunk.strip():
                    chunks.append({'content': sub_chunk.strip(), 'metadata': metadata})
                    
        print_progress_bar(3, 3, prefix='Splitting text...     ', suffix='Split completed ', length=30)
        return chunks
