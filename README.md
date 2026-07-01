# 👁️ Computer Vision (CV) Pipeline Architecture

This module forms the core of the document extraction and visual understanding pipeline. It is specifically designed to parse visually complex academic papers or documents, extracting both textual data and visual elements (such as tables, figures, and mathematical formulas), and bridging the gap between them using advanced Vision-Language Models (VLMs).

## 🔄 Pipeline Workflow (4-Phase Architecture)

The CV pipeline is orchestrated by `cv_orchestrator.py` and operates in four sequential phases:

### Phase 1: Document Layout Parsing (`layout_parser.py`)
- **Objective:** Analyze the spatial layout of PDF pages to identify and isolate specific document regions.
- **Process:** The system converts PDF pages into high-resolution images. An object detection model scans the page to draw bounding boxes around key elements, classifying them as `Title`, `Text`, `Image`, `Table`, or `Formula`. Non-text regions are spatially cropped and saved as independent image assets, while text regions undergo spatial text extraction. Overlapping boxes are intelligently merged to maintain structural integrity.

### Phase 2: Semantic Text Chunking (`text_chunker.py`)
- **Objective:** Structure and segment the extracted raw text into meaningful chunks for downstream NLP processing.
- **Process:** The text is first formatted into standard Markdown using the recognized `Title` headers. It is then recursively split based on Markdown headers and character limits, ensuring that token sizes are optimal for vector databases. Metadata, such as the originating page numbers, is injected into each chunk.

### Phase 3: Visual Semantic Processing (`visual_processor.py`)
- **Objective:** Translate visual assets (cropped in Phase 1) into rich, semantic textual representations.
- **Process:** The cropped images of tables, figures, and formulas are passed to a Multimodal Large Language Model (VLM). Based on targeted prompts, the VLM performs:
  - **Dense Captioning:** Generates comprehensive narrative descriptions for architectural diagrams and figures.
  - **Table Reconstruction:** Accurately transcribes image-based tables into pure Markdown table formats.
  - **Formula Transcription:** Converts mathematical equations into standard LaTeX syntax.

### Phase 4: Cross-Modal Sequence Binding (`context_binder.py`)
- **Objective:** Unify the disparate textual and visual modalities into a single, cohesive document.
- **Process:** The binder scans the text chunks for visual references (e.g., *"As shown in Figure 1"* or *"Table 2"*). It cross-references these citations with the semantic narratives generated in Phase 3 and seamlessly embeds the VLM's descriptions, markdown tables, or LaTeX formulas back into the logical reading order of the Markdown document. The final output is serialized as both a `.md` file and a `.jsonl` structured corpus.

---

## 🛠️ Technical Stack & Libraries

The CV module is built upon a robust stack of high-performance libraries:

- **PyMuPDF (`fitz`):** Utilized for high-speed PDF parsing, pixel rendering, and precise spatial clipping (cropping) of visual elements based on bounding box coordinates.
- **Ultralytics / DocLayout-YOLO:** The backbone for Object Detection. It provides the deep learning framework to detect complex, dense document layouts efficiently.
- **Llama.cpp (`llama_cpp_python`):** Serves as the inference engine for executing quantized Large Language Models locally. It specifically utilizes multimodal chat handlers (`Qwen3VLChatHandler`, `Llava15ChatHandler`) to process image-text inputs seamlessly.
- **LangChain (`langchain_text_splitters`):** Powers the hierarchical and recursive text chunking mechanism, ensuring context-aware segmentation based on Markdown headers.
- **NumPy & OpenCV:** Used implicitly for matrix operations and image color space conversions (e.g., RGBA to RGB) prior to YOLO inference.

---

## 🤖 Models Utilized

1. **Document Layout Analysis (DLA) Model:**
   - **Primary:** `DocLayout-YOLO` (specifically the `doclayout_yolo-layout-sm.pt` variant from HuggingFace). This model is highly optimized for recognizing structured and unstructured document elements using a global-to-local receptive mechanism.
   - **Fallback:** Standard `YOLOv10` or `YOLOv8` architectures if the primary model is unavailable.

2. **Vision-Language Model (VLM):**
   - **Primary:** **Qwen-VL** (e.g., Qwen2.5-VL or Qwen3-VL), running in a quantized format (GGUF) via `llama.cpp`. It acts as the cognitive engine to understand complex charts and transcribe tables/formulas.
   - **Fallback:** **LLaVA 1.5**, utilized as an alternative multimodal model to guarantee robust visual reasoning capabilities.
