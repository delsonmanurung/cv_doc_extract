import sys
import os
import time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import QWEN_VL_MODEL_PATH, QWEN_VL_MMPROJ_PATH, DATA_DIR, LLAMA_N_CTX, LLAMA_N_THREADS, LLAMA_N_GPU_LAYERS
from src.cv.layout_parser import LayoutParser
from src.cv.text_chunker import TextChunker
from src.cv.visual_processor import VisualProcessor
from src.cv.context_binder import ContextBinder
import re

def get_short_dir_name(pdf_path: str, max_len=30) -> str:
    stem = Path(pdf_path).stem
    clean_stem = re.sub('[^a-zA-Z0-9_\\-]', '', stem)
    return clean_stem[:max_len]

def run_phase1_pipeline(pdf_path: str):
    print(f'\n==================================================')
    print(f'Pipeline of: {Path(pdf_path).name}')
    print(f'==================================================')
    pipeline_start = time.time()
    short_name = get_short_dir_name(pdf_path)
    output_dir = os.path.join('data', 'processed', short_name)
    cropped_dir = os.path.join(output_dir, 'cropped')
    result_dir = os.path.join(output_dir, 'result')
    phase1_output_path = os.path.join(result_dir, f'{short_name}_phase1_blocks.json')
    blocks = []
    if os.path.exists(phase1_output_path):
        print(f'Loaded cached layout from: {Path(phase1_output_path).name}')
        import json
        from config.models import ParsedBlock, BoundingBox
        with open(phase1_output_path, 'r', encoding='utf-8') as f:
            blocks_data = json.load(f)
            blocks = [ParsedBlock(block_id=b['id'], block_type=b['type'], content=b['content'], image_path=b['image_path'], reading_order=b['reading_order'], page_number=b['page'], bbox=BoundingBox(x0=0, y0=0, x1=0, y1=0)) for b in blocks_data]
        phase1_elapsed = 0.0
        print(f'Phase 1 elapsed time: 0.00s (Loaded from cache)')
    else:
        if os.path.exists(output_dir):
            import shutil
            shutil.rmtree(output_dir)
        os.makedirs(cropped_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)
        print(f'Directory initialized for visual assets: {cropped_dir}/')
        print(f'Directory initialized for analytical results: {result_dir}/')
        print('\n--- Phase 1: Layout Parsing ---')
        phase1_start = time.time()
        parser = LayoutParser()
        blocks = parser.parse(pdf_path, output_dir=cropped_dir)
        print(f'Extracted {len(blocks)} blocks and established reading order.')
        import json
        with open(phase1_output_path, 'w', encoding='utf-8') as f:
            blocks_dict = [{'id': b.block_id, 'type': b.block_type, 'content': b.content, 'image_path': b.image_path, 'reading_order': b.reading_order, 'page': b.page_number} for b in blocks]
            json.dump(blocks_dict, f, indent=2, ensure_ascii=False)
        phase1_elapsed = time.time() - phase1_start
        print(f'Phase 1 execution successful. Raw layout extraction metrics saved to: {Path(phase1_output_path).name}')
        print(f'Phase 1 elapsed time: {phase1_elapsed:.2f}s')
    print('\n--- Phase 2: Semantic Text Chunking ---')
    phase2_start = time.time()
    chunker = TextChunker()
    chunks = chunker.chunk_document(blocks)
    phase2_output_path = os.path.join(result_dir, f'{short_name}_phase2_chunks.json')
    with open(phase2_output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    phase2_elapsed = time.time() - phase2_start
    print(f'Phase 2 execution successful. Chunked semantic dataset saved to: {Path(phase2_output_path).name}')
    print(f'Phase 2 elapsed time: {phase2_elapsed:.2f}s')
    print('\n--- Verification: Phase 2 Output Validation ---')
    if chunks:
        is_valid = all(('content' in c and 'metadata' in c and ('pages' in c['metadata']) for c in chunks))
        if is_valid:
            print('Validation Passed: Schema coherence established across all chunk nodes.')
        else:
            print('Validation Failed: Sub-optimal metadata alignment detected in chunk nodes.')
        print(f'Processed Entity Count (Chunks): {len(chunks)}')
    else:
        print('Error Exception: Execution halted due to an absence of generated text chunks.')
    print('\n--- Phase 3: Visual Semantic Processing (VLM Analysis) ---')
    phase3_start = time.time()
    visual_processor = VisualProcessor(model_path=str(QWEN_VL_MODEL_PATH), mmproj_path=str(QWEN_VL_MMPROJ_PATH), n_ctx=LLAMA_N_CTX, n_threads=LLAMA_N_THREADS, n_gpu_layers=LLAMA_N_GPU_LAYERS)
    visual_results = visual_processor.analyze_blocks(blocks)
    phase3_output_path = os.path.join(result_dir, f'{short_name}_phase3_visual_results.json')
    with open(phase3_output_path, 'w', encoding='utf-8') as f:
        visual_results_dict = [{'block_id': r.block_id, 'narrative': r.narrative} for r in visual_results]
        json.dump(visual_results_dict, f, indent=2, ensure_ascii=False)
    phase3_elapsed = time.time() - phase3_start
    print(f'Phase 3 execution successful. Visual analysis corpus saved to: {Path(phase3_output_path).name}')
    print(f'Phase 3 elapsed time: {phase3_elapsed:.2f}s')
    print('\n--- Phase 4: Cross-Modal Sequence Binding ---')
    phase4_start = time.time()
    binder = ContextBinder()
    unified_markdown = binder.bind_and_sequence(blocks, visual_results)
    output_md_path = os.path.join(result_dir, f'{short_name}_final.md')
    output_jsonl_path = os.path.join(result_dir, f'{short_name}_final.jsonl')
    with open(output_md_path, 'w', encoding='utf-8') as f:
        f.write(unified_markdown)
    binder.export_to_jsonl(blocks, visual_results, output_jsonl_path)
    phase4_elapsed = time.time() - phase4_start
    print(f'Phase 4 elapsed time: {phase4_elapsed:.2f}s')
    total_elapsed = time.time() - pipeline_start
    print('\n--- Pipeline Execution Completed Successfully ---')
    print(f'Consolidated Markdown serialized to: {Path(output_md_path).name}')
    print(f'JSONL structured corpus written to: {Path(output_jsonl_path).name}')
    print('\n' + '=' * 50)
    print('Execution Time Summary:')
    print(f'- Phase 1 (Layout Parsing): {phase1_elapsed:.2f}s')
    print(f'- Phase 2 (Semantic Text Chunking): {phase2_elapsed:.2f}s')
    print(f'- Phase 3 (Visual Semantic Processing): {phase3_elapsed:.2f}s')
    print(f'- Phase 4 (Cross-Modal Sequence Binding): {phase4_elapsed:.2f}s')
    print(f'- Total Pipeline Time: {total_elapsed:.2f}s')
    print('=' * 50)

def main():
    raw_dir = DATA_DIR / 'raw'
    pdf_files = list(raw_dir.glob('*.pdf'))
    if not pdf_files:
        print('No input datasets identified in data/raw. Initiating fallback diagnostic procedure.')
        test_pdf = str(raw_dir / 'sample_paper.pdf')
        short_name = get_short_dir_name(test_pdf)
        output_md_path = DATA_DIR / 'processed' / short_name / 'result' / f'{short_name}_final.md'
        output_jsonl_path = DATA_DIR / 'processed' / short_name / 'result' / f'{short_name}_final.jsonl'
        if output_md_path.exists() and output_jsonl_path.exists():
            print(f"\nDiagnostic procedure bypassed. Target artifact '{test_pdf}' possesses valid analytical cache.")
        else:
            run_phase1_pipeline(test_pdf)
    else:
        print(f'Discovered {len(pdf_files)} PDF dataset(s) scheduled for processing.')
        for pdf_file in pdf_files:
            short_name = get_short_dir_name(str(pdf_file))
            output_md_path = DATA_DIR / 'processed' / short_name / 'result' / f'{short_name}_final.md'
            output_jsonl_path = DATA_DIR / 'processed' / short_name / 'result' / f'{short_name}_final.jsonl'
            if output_md_path.exists() and output_jsonl_path.exists():
                print(f"\nProcessing skipped. Dataset '{pdf_file.name}' has already been processed.")
            else:
                run_phase1_pipeline(str(pdf_file))
    os._exit(0)

if __name__ == '__main__':
    main()
