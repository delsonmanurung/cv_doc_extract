from pydantic import BaseModel
from typing import Optional

class BoundingBox(BaseModel):
    x0: float; y0: float; x1: float; y1: float

class ParsedBlock(BaseModel):
    block_id: str
    block_type: str
    content: Optional[str] = None
    image_path: Optional[str] = None
    reading_order: int = 0
    page_number: int
    bbox: BoundingBox

class VisualAnalysisResult(BaseModel):
    block_id: str
    narrative: str
    semantic_id: str = ""
