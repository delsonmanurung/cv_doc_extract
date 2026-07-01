import re
from typing import List, Dict
from config.models import ParsedBlock, VisualAnalysisResult
import json

class ContextBinder:
    def __init__(self):
        self.ref_pattern = re.compile(r'(?i)(fig\b\.?|figure|table|eq\b\.?|equation)\s*(\d+)')

    def bind_and_sequence(self, blocks: List[ParsedBlock], visual_results: List[VisualAnalysisResult]) -> str:
        visual_map: Dict[str, Dict[str, str]] = {
            res.block_id: {'narrative': res.narrative, 'semantic_id': res.semantic_id} 
            for res in visual_results
        }
         
        ref_to_block_id = {}
        for block in blocks:
            if block.block_type in ['Image', 'Table', 'Formula']:
                v_data = visual_map.get(block.block_id, {})
                sem_id = v_data.get('semantic_id', '')
                
                parsed = re.search(r'(?i)(fig|figure|table|eq|equation)\s*(\d+)', sem_id)
                if parsed:
                    ref_type = 'image' if 'fig' in parsed.group(1).lower() else ('table' if 'table' in parsed.group(1).lower() else 'formula')
                    ref_to_block_id[(ref_type, parsed.group(2))] = block.block_id
                else:
                    nums = re.findall(r'\d+', block.block_id)
                    if nums:
                        num = nums[0]
                        ref_type = 'image' if block.block_type == 'Image' else ('table' if block.block_type == 'Table' else 'formula')
                        if (ref_type, num) not in ref_to_block_id:
                            ref_to_block_id[(ref_type, num)] = block.block_id

        markdown_lines = []
        for block in blocks:
            if block.block_type == 'Text':
                text_content = block.content or ''
                markdown_lines.append(f'{text_content}\n')
                
                matches = self.ref_pattern.findall(text_content)
                for prefix, num in matches:
                    prefix_lower = prefix.lower()
                    if 'fig' in prefix_lower:
                        ref_type, display_prefix = 'image', 'Fig'
                    elif 'table' in prefix_lower:
                        ref_type, display_prefix = 'table', 'Table'
                    else:
                        ref_type, display_prefix = 'formula', 'Eq'
                        
                    matched_block_id = ref_to_block_id.get((ref_type, num))
                    if matched_block_id and matched_block_id in visual_map:
                        narrative = visual_map[matched_block_id]['narrative']
                        markdown_lines.append(f'\n> **[Visual Context: {display_prefix} {num}]**: \n{narrative}\n')
                        
            elif block.block_type in ['Image', 'Table', 'Formula']:
                v_data = visual_map.get(block.block_id, {})
                narrative = v_data.get('narrative', f'[No description available]')
                sem_id = v_data.get('semantic_id', block.block_id)
                 
                markdown_lines.append(f'\n\n**[{block.block_type}: {sem_id or block.block_id}]**')
                markdown_lines.append(f'{narrative}\n\n')
                
        return '\n'.join(markdown_lines)

    def export_to_jsonl(self, blocks: List[ParsedBlock], visual_results: List[VisualAnalysisResult], output_path: str):
        visual_map: Dict[str, str] = {res.block_id: res.narrative for res in visual_results}
        
        current_headers = {}
        with open(output_path, 'w', encoding='utf-8') as f:
            for block in blocks:
                if block.block_type == 'Title' and block.content:
                    title = block.content.strip()
                    if re.match(r'^\d+\.\d+\.\d+\s+', title):
                        current_headers['Header 4'] = title
                    elif re.match(r'^\d+\.\d+\s+', title):
                        current_headers['Header 3'] = title
                        current_headers.pop('Header 4', None)
                    elif re.match(r'^\d+\s+', title):
                        current_headers['Header 2'] = title
                        current_headers.pop('Header 3', None)
                        current_headers.pop('Header 4', None)
                    else:
                        current_headers['Header 1'] = title
                        current_headers.pop('Header 2', None)
                        current_headers.pop('Header 3', None)
                        current_headers.pop('Header 4', None)

                data = {
                    'block_id': block.block_id, 
                    'block_type': block.block_type, 
                    'page_number': block.page_number, 
                    'reading_order': block.reading_order,
                    'headers': dict(current_headers)
                }
                if block.block_type == 'Text':
                    data['content'] = block.content
                else:
                    data['narrative'] = visual_map.get(block.block_id, '')
                    data['image_path'] = block.image_path
                f.write(json.dumps(data, ensure_ascii=False) + '\n')