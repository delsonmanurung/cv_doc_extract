import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

YOLO_MODEL_PATH = MODELS_DIR / "doclayout_yolo.pt"
QWEN_VL_MODEL_PATH = MODELS_DIR / "Qwen3-VL-2B-Instruct-Q4_K_M.gguf"
QWEN_VL_MMPROJ_PATH = MODELS_DIR / "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf"
QWEN_LLM_MODEL_PATH = MODELS_DIR / "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
BGE_M3_MODEL_PATH = MODELS_DIR / "bge-m3-q4_k_m.gguf"

if not QWEN_VL_MODEL_PATH.exists():
    print(f"Warning: Model not found at {QWEN_VL_MODEL_PATH}")
if not QWEN_VL_MMPROJ_PATH.exists():
    print(f"Warning: Multimodal Projector model not found at {QWEN_VL_MMPROJ_PATH}")

LLAMA_N_CTX = 4096
LLAMA_N_THREADS = 8
LLAMA_N_GPU_LAYERS = -1

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "mulvi_papers"

DATA_DIR.mkdir(exist_ok=True)
(DATA_DIR / "raw").mkdir(exist_ok=True)
(DATA_DIR / "processed").mkdir(exist_ok=True)
