from pathlib import Path
from config.models import ParsedBlock, BoundingBox
from typing import List
import os
import re
import sys
try:
    import fitz
    try:
        from doclayout_yolo import YOLOv10 as YOLO
    except ImportError:
        from ultralytics import YOLO
    import numpy as np
    from PIL import Image
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f'Warning: Missing dependencies (ultralytics, pymupdf). Error: {e}')
    DEPENDENCIES_AVAILABLE = False
from config.settings import YOLO_MODEL_PATH

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=30, fill='█'):
    percent = ('{0:.' + str(decimals) + 'f}').format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    if iteration == total:
        print()

class LayoutParser:

    def __init__(self):
        self.yolo_model = None
        if DEPENDENCIES_AVAILABLE:
            try:
                import torch
                _original_load = torch.load

                def _patched_load(*args, **kwargs):
                    kwargs['weights_only'] = False
                    return _original_load(*args, **kwargs)
                torch.load = _patched_load
                if not os.path.exists(YOLO_MODEL_PATH):
                    print(f'Model not found at {YOLO_MODEL_PATH}. Using a default YOLO layout model from HuggingFace.')
                    model_path = 'hf://opendatalab/DocLayout-YOLO/doclayout_yolo-layout-sm.pt'
                else:
                    model_path = str(YOLO_MODEL_PATH)
                self.yolo_model = YOLO(model_path)
                try:
                    import doclayout_yolo.nn.modules.g2l_crm as g2l_crm
                    import torch.nn.functional as F

                    def patched_dilated_conv(self, x, dilation):
                        act = getattr(self.dcv, 'act', torch.nn.Identity())
                        bn = getattr(self.dcv, 'bn', torch.nn.Identity())
                        weight = self.dcv.conv.weight
                        padding = dilation * (self.k // 2)
                        return act(bn(F.conv2d(x, weight, stride=1, padding=padding, dilation=dilation)))
                    g2l_crm.DilatedBlock.dilated_conv = patched_dilated_conv
                except Exception as ex:
                    print(f'Warning: Could not monkey-patch DilatedBlock: {ex}')
                if hasattr(self.yolo_model, 'model') and self.yolo_model.model is not None:
                    self.yolo_model.model.fused = True
                torch.load = _original_load
            except Exception as e:
                raise RuntimeError(f'Failed to initialize DocLayout-YOLO: {e}')

    def detect_layout(self, img_np):
        return self.yolo_model(img_np, verbose=False)[0]

    def extract_text_spatially(self, page, pdf_rect) -> str:
        return page.get_text('text', clip=pdf_rect).strip()

    def crop_visual_block(self, page, pdf_rect, image_path: str):
        crop_pix = page.get_pixmap(clip=pdf_rect, dpi=200)
        crop_pix.save(image_path)

    def _merge_boxes(self, boxes: List[dict], y_threshold=50, x_threshold=50) -> List[dict]:
        """Merges closely adjacent bounding boxes of the same type (Table or Image)."""
        merged = True
        while merged:
            merged = False
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    b1, b2 = boxes[i], boxes[j]
                    if b1['type'] == b2['type'] and b1['type'] in ['Table', 'Image']:
                        x_overlap = max(0, min(b1['x2'], b2['x2']) - max(b1['x1'], b2['x1']))
                        y_overlap = max(0, min(b1['y2'], b2['y2']) - max(b1['y1'], b2['y1']))
                        
                        x_dist = max(0, max(b1['x1'], b2['x1']) - min(b1['x2'], b2['x2']))
                        y_dist = max(0, max(b1['y1'], b2['y1']) - min(b1['y2'], b2['y2']))

                        if (x_overlap > 0 and y_dist < y_threshold) or \
                           (y_overlap > 0 and x_dist < x_threshold) or \
                           (x_overlap > 0 and y_overlap > 0):
                            
                            new_box = {
                                'type': b1['type'],
                                'x1': min(b1['x1'], b2['x1']),
                                'y1': min(b1['y1'], b2['y1']),
                                'x2': max(b1['x2'], b2['x2']),
                                'y2': max(b1['y2'], b2['y2'])
                            }
                            boxes[i] = new_box
                            del boxes[j]
                            merged = True
                            break
                if merged:
                    break
        return boxes

    def parse(self, pdf_path: str, output_dir: str='data/processed') -> List[ParsedBlock]:
        blocks = []
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError('Missing dependencies (ultralytics, pymupdf). Please install them.')
        if self.yolo_model is None:
            raise RuntimeError('YOLO model is not initialized.')
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f'PDF file not found: {pdf_path}')
        try:
            doc = fitz.open(pdf_path)
            processed_dir = Path(output_dir)
            processed_dir.mkdir(parents=True, exist_ok=True)
            block_counter = 1
            paper_stem = Path(pdf_path).stem
            total_pages = len(doc)
            for page_num in range(total_pages):
                print_progress_bar(page_num + 1, total_pages, prefix='Parsing layout...', suffix=f'Page {page_num + 1}/{total_pages}', length=30)
                page = doc[page_num]
                pix = page.get_pixmap(dpi=150)
                img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                if pix.n == 4:
                    import cv2
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
                
                results = self.detect_layout(img_np)
                class_names = results.names
                
                raw_boxes = []
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    class_name = class_names[cls_id].lower()
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    if conf < 0.4 or 'abandon' in class_name or 'header' in class_name or ('footer' in class_name):
                        continue
                    
                    img_h = img_np.shape[0]
                    if (y2 / img_h) < 0.08:
                        continue
                    if (y1 / img_h) > 0.92:
                        continue
                    
                    if 'title' in class_name:
                        mapped_type = 'Title'
                    elif 'table' in class_name:
                        mapped_type = 'Table'
                    elif 'formula' in class_name:
                        mapped_type = 'Formula'
                    elif 'figure' in class_name or 'image' in class_name:
                        mapped_type = 'Image'
                    else:
                        mapped_type = 'Text'
                    
                    if mapped_type in ['Table', 'Image', 'Formula']:
                        height = y2 - y1
                        y2 = min(img_h, y2 + height * 0.15) 

                    raw_boxes.append({
                        'type': mapped_type,
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
                    })

                merged_boxes = self._merge_boxes(raw_boxes, y_threshold=50, x_threshold=50)

                page_blocks = []
                for bbox_data in merged_boxes:
                    mapped_type = bbox_data['type']
                    x1, y1, x2, y2 = bbox_data['x1'], bbox_data['y1'], bbox_data['x2'], bbox_data['y2']
                    
                    scale = 72.0 / 150.0
                    pdf_rect = fitz.Rect(x1 * scale, y1 * scale, x2 * scale, y2 * scale)
                    block_id = f'{mapped_type.lower()}_{block_counter}'
                    block_counter += 1
                    
                    content = None
                    image_path = None
                    
                    if mapped_type in ['Text', 'Title']:
                        content = self.extract_text_spatially(page, pdf_rect)
                        if not content:
                            continue
                    elif mapped_type in ['Image', 'Table', 'Formula']: 
                        image_file_name = f'{paper_stem}_{block_id}.png'
                        image_path = str(processed_dir / image_file_name)
                        self.crop_visual_block(page, pdf_rect, image_path)
                        content = f'[Visual element cropped to {image_path}]'
                        
                    page_blocks.append(ParsedBlock(
                        block_id=block_id, 
                        block_type=mapped_type, 
                        content=content, 
                        image_path=image_path,
                        reading_order=0,
                        bbox=BoundingBox(x0=x1 * scale, y0=y1 * scale, x1=x2 * scale, y1=y2 * scale), 
                        page_number=page_num + 1
                    ))
                blocks.extend(page_blocks)
        except Exception as err:
            raise RuntimeError(f'Error parsing PDF with DocLayout-YOLO/PyMuPDF: {err}')
        blocks.sort(key=lambda b: (b.page_number, b.bbox.x0 // 200, b.bbox.y0))
        filtered_blocks = []
        for block in blocks:
            if block.block_type in ['Text', 'Title'] and block.content:
                text = block.content.strip().lower()
                normalized_text = re.sub('^\\d+[\\.\\s]+', '', text)
                normalized_text = re.sub('[^\\w\\s]', '', normalized_text).strip()
                if normalized_text in ['references', 'bibliography']:
                    print(f"Detected bibliography section: '{block.content}' (Block ID: {block.block_id}). Stopping layout extraction.")
                    break
            filtered_blocks.append(block)
        for i, block in enumerate(filtered_blocks):
            block.reading_order = i + 1
        return filtered_blocks