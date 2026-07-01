from config.models import ParsedBlock, VisualAnalysisResult
from llama_cpp import Llama
from typing import List
import base64
import os
import sys
import contextlib
import re

@contextlib.contextmanager
def suppress_stdout_stderr():
    null_fds = []
    save_fds = []
    try:
        null_fds.append(os.open(os.devnull, os.O_RDWR))
        null_fds.append(os.open(os.devnull, os.O_RDWR))
        save_fds.append(os.dup(1))
        save_fds.append(os.dup(2))
        os.dup2(null_fds[0], 1)
        os.dup2(null_fds[1], 2)
        yield
    except Exception:
        yield
    finally:
        if save_fds:
            os.dup2(save_fds[0], 1)
            os.dup2(save_fds[1], 2)
        for fd in null_fds + save_fds:
            try:
                os.close(fd)
            except OSError:
                pass

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=30, fill='█'):
    percent = ('{0:.' + str(decimals) + 'f}').format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    if iteration == total:
        print()

class VisualProcessor:

    def __init__(self, model_path: str, mmproj_path: str, n_ctx: int=4096, n_threads: int=8, n_gpu_layers: int=-1):
        self.llm = None
        self.chat_handler = None
        try:
            chat_format = 'chatml'
            try:
                from llama_cpp.llama_chat_format import Qwen3VLChatHandler
                self.chat_handler = Qwen3VLChatHandler(clip_model_path=mmproj_path)
                chat_format = 'qwen3-vl'
                print('Multimodal Qwen3VLChatHandler loaded.')
            except ImportError:
                try:
                    from llama_cpp.llama_chat_format import Qwen25VLChatHandler
                    self.chat_handler = Qwen25VLChatHandler(clip_model_path=mmproj_path)
                    chat_format = 'qwen2.5-vl'
                    print('Fallback Multimodal Qwen25VLChatHandler loaded.')
                except ImportError:
                    try:
                        from llama_cpp.llama_chat_format import Llava15ChatHandler
                        self.chat_handler = Llava15ChatHandler(clip_model_path=mmproj_path)
                        chat_format = 'chatml'
                        print('Fallback Multimodal Llava15ChatHandler loaded.')
                    except ImportError:
                        print('Multimodal handlers not found. Proceeding with standard LLM load.')
            with suppress_stdout_stderr():
                self.llm = Llama(
                    model_path=model_path,
                    chat_handler=self.chat_handler,
                    chat_format=chat_format if self.chat_handler else 'chatml',
                    n_ctx=n_ctx,
                    n_threads=n_threads,
                    n_gpu_layers=n_gpu_layers,
                    verbose=False
                )
            print('Loaded VLM model successfully.')
        except Exception as e:
            print(f'Warning: Could not load VLM model. Exception: {e}')

    def analyze_blocks(self, blocks: List[ParsedBlock]) -> List[VisualAnalysisResult]:
        results = []
        visual_blocks = [b for b in blocks if b.block_type in ['Image', 'Table', 'Formula'] and b.image_path]
        total_visual = len(visual_blocks)
        
        if total_visual == 0:
            print('No visual blocks found to analyze.')
            return results
            
        for idx, block in enumerate(visual_blocks):
            print_progress_bar(idx + 1, total_visual, prefix='Analyzing visuals... ', suffix=f'Block {idx + 1}/{total_visual} ({block.block_id})', length=30)
            
            narrative, semantic_id = self._generate_narrative(block)
            narrative = self._clean_leakage(narrative)
            
            results.append(VisualAnalysisResult(
                block_id=block.block_id, 
                narrative=narrative, 
                semantic_id=semantic_id
            ))
        return results

    def _clean_leakage(self, text: str) -> str:
        return re.sub(r'^(facts|Fact|FACTS)\s*\n?', '', text, flags=re.IGNORECASE).strip()
    
    def _image_to_base64_data_uri(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            return ''
        ext = os.path.splitext(file_path)[1].lower()
        mime_type = 'image/png' if ext == '.png' else 'image/jpeg'
        with open(file_path, 'rb') as img_file:
            base64_data = base64.b64encode(img_file.read()).decode('utf-8')
        return f'data:{mime_type};base64,{base64_data}'

    def _generate_narrative(self, block: ParsedBlock) -> tuple[str, str]:
        if not self.llm:
            return f'[Mocked Visual Description for {block.block_id}]', ''
            
        data_uri = self._image_to_base64_data_uri(block.image_path)
        if not data_uri:
             return f'[Image file not found for {block.block_id}]', ''
             
        try:
            if block.block_type == 'Table':
                prompt_text = (
                    "Identify the table number (e.g., 'Table 1') from the image or caption. "
                    "Then, convert the table content into a clean Markdown Table format. "
                    "CRITICAL RULES: "
                    "1. Start your response EXACTLY with: [ID: Table X]\n (replace X with the number). "
                    "2. Output ONLY the Markdown table after the ID. "
                    "3. DO NOT write any narrative or introductory phrases."
                )
            elif block.block_type == 'Formula':
                prompt_text = (
                    "Extract the mathematical formula in this image into standard LaTeX format enclosed in $$ $$. "
                    "Identify the equation number if visible (e.g., '(1)', 'Eq. 2'). "
                    "CRITICAL RULES: "
                    "1. Start your response EXACTLY with: [ID: Eq. X]\n or [ID: Unknown]\n. "
                    "2. Output ONLY the LaTeX code after the ID. DO NOT explain the formula."
                )
            else:
                prompt_text = (
                    "Identify the figure number (e.g., 'Fig. 1', 'Figure 2') from the image or caption. "
                    "Then, provide a dense, factual description of the diagram/chart. "
                    "CRITICAL RULES: "
                    "1. Start your response EXACTLY with: [ID: Fig. X]\n (replace X with the number). "
                    "2. DO NOT use introductory phrases like 'This image shows'."
                )

            messages = [{'role': 'user', 'content': [{'type': 'text', 'text': prompt_text}, {'type': 'image_url', 'image_url': {'url': data_uri}}]}]
            
            with suppress_stdout_stderr():
                response = self.llm.create_chat_completion(
                    messages=messages, 
                    max_tokens=800,
                    temperature=0.1,
                    top_p=0.9, 
                    repeat_penalty=1.15
                )
            
            raw_output = response['choices'][0]['message']['content'].strip()
            semantic_id = ""
            id_match = re.search(r'\[ID:\s*(.*?)\]', raw_output)
            if id_match:
                semantic_id = id_match.group(1).strip()
                narrative = raw_output[id_match.end():].strip()
            else:
                narrative = raw_output

            return narrative, semantic_id

        except Exception as e:
            print(f'Error generating narrative: {e}')
            return f'[Failed to analyze {block.block_id}]', ''