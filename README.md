# Imagereasoning

![Language](https://img.shields.io/badge/Language-Python-3776AB?style=flat-square) ![Stars](https://img.shields.io/github/stars/Devanik21/ImageReasoning?style=flat-square&color=yellow) ![Forks](https://img.shields.io/github/forks/Devanik21/ImageReasoning?style=flat-square&color=blue) ![Author](https://img.shields.io/badge/Author-Devanik21-black?style=flat-square&logo=github) ![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

> Visual question answering and image reasoning — multimodal AI that understands images and answers natural language questions about them.

---

**Topics:** `image-understanding` · `cognitive-architecture` · `deep-reinforcement-learning` · `gemini-2-5` · `reward-scoring` · `step-wise-reasoning` · `student-teacher-framework` · `visual-reasoning`

## Overview

ImageReasoning is a multimodal AI application that combines computer vision and natural language
understanding to answer free-form questions about images — a task known as Visual Question Answering (VQA).
It integrates state-of-the-art vision-language models (BLIP-2, LLaVA, or GPT-4 Vision) to handle
questions ranging from simple object recognition ('what colour is the car?') to complex spatial
reasoning ('which object is between the chair and the window?') to abstract inference ('what is the
person in the image likely feeling?').

The application provides two distinct modes. In the standard VQA mode, users upload an image and
ask any natural language question, receiving a direct answer with an attention map overlay showing
which image regions the model focused on. In the visual reasoning chain mode, the model is prompted
to reason step by step before answering — first identifying relevant objects, then their relationships,
then inferring the answer — providing an interpretable reasoning trace that makes complex spatial
and causal questions more accurate and verifiable.

A benchmark evaluation module allows systematic comparison of different VLMs (BLIP-2, LLaVA-1.5,
GPT-4V) on standard VQA datasets (VQA v2, GQA, TextVQA) with configurable question categories,
providing reproducible performance metrics for model selection in specific visual reasoning tasks.

---

## Motivation

Vision-language understanding is one of the most practically powerful capabilities of modern AI —
enabling applications from medical image Q&A to accessibility tools for visually impaired users
to autonomous systems that can answer questions about their visual field. Making these capabilities
accessible through a clean, well-documented application reduces the barrier for researchers and
developers to experiment with and build on multimodal AI.

---

## Architecture

```
Image Input + Natural Language Question
        │
  Vision Encoder (ViT-L/14 or CLIP)
  ├── Patch embeddings: (224×224) → 196 tokens
  └── [CLS] token: global image representation
        │
  Querying Transformer (BLIP-2) or
  Visual Instruction Tuning (LLaVA)
        │
  Language Model Decoder
  (OPT-6.7B / Vicuna-13B / GPT-4)
        │
  Free-form answer generation
        │
  Optional: attention map visualisation
           chain-of-thought reasoning trace
```

---

## Features

### Free-Form Visual Question Answering
Answer any natural language question about an uploaded image using BLIP-2 or LLaVA — open-ended, multiple choice, yes/no, counting, and colour/attribute questions all supported.

### Visual Reasoning Chain
Chain-of-thought visual reasoning mode prompts the model to explicitly state its reasoning steps before answering, improving accuracy on spatial relationship and multi-hop inference questions.

### Attention Map Visualisation
Grad-CAM or attention rollout overlay on the input image showing which spatial regions most influenced the model's answer — making VQA predictions interpretable.

### GPT-4 Vision Integration
Optional GPT-4V backend for highest-accuracy complex reasoning questions — particularly useful for reading text in images (OCR-VQA) and detailed scene understanding.

### Multi-Image Comparison Mode
Upload two images and ask comparison questions ('which image has more people?', 'how do these two charts differ?') using a multi-image prompt template.

### Document and Chart Q&A
Specialised mode for structured images: tables, charts, diagrams, and document pages — using layout-aware processing for accurate numerical and relationship extraction.

### VQA Benchmark Evaluation
Systematic evaluation on VQA v2, GQA, and TextVQA benchmark subsets with per-category accuracy breakdown (yes/no, number, other) and model comparison table.

### Batch Processing Pipeline
Process a JSONL file of image-question pairs for bulk VQA inference, with parallel execution and results CSV export.

---

## Tech Stack

| Library / Tool | Role | Why This Choice |
|---|---|---|
| **BLIP-2 (Salesforce)** | Primary VLM | Bootstrapped Language-Image Pre-training for efficient VQA |
| **LLaVA (optional)** | Visual instruction model | Large Language and Vision Assistant for complex reasoning |
| **OpenAI GPT-4V (optional)** | Highest accuracy | GPT-4 Vision for complex and text-in-image questions |
| **Transformers (HuggingFace)** | Model loading | Unified interface for BLIP-2, LLaVA, and CLIP models |
| **PyTorch** | Deep learning backend | Model inference and attention extraction |
| **Pillow / OpenCV** | Image processing | Upload handling, preprocessing, attention overlay |
| **Streamlit** | Application UI | Image upload, question input, answer display with overlay |

---

## Getting Started

### Prerequisites

- Python 3.9+ (or Node.js 18+ for TypeScript/JavaScript projects)
- A virtual environment manager (`venv`, `conda`, or equivalent)
- API keys as listed in the Configuration section

### Installation

```bash
git clone https://github.com/Devanik21/ImageReasoning.git
cd ImageReasoning
python -m venv venv && source venv/bin/activate
pip install streamlit transformers torch pillow accelerate
# BLIP-2 weights download automatically from HuggingFace on first run (~5GB for full model)
# Optional: GPU recommended for <3s inference
echo 'OPENAI_API_KEY=sk-...' > .env  # only if using GPT-4V backend
streamlit run app.py
```

---

## Usage

```bash
# Launch VQA interface
streamlit run app.py

# Single question from CLI
python ask.py --image photo.jpg --question 'How many people are in this image?'

# Batch VQA evaluation
python batch_vqa.py --input questions.jsonl --output answers.csv --model blip2

# Benchmark evaluation
python benchmark.py --dataset vqa_v2 --split val --model blip2 --samples 1000
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `DEFAULT_MODEL` | `blip2-opt-2.7b` | VLM model: blip2-opt-2.7b, blip2-flan-t5-xl, llava-1.5-7b |
| `OPENAI_API_KEY` | `(optional)` | For GPT-4 Vision backend |
| `DEVICE` | `auto` | Inference device: auto, cpu, cuda, mps |
| `ATTENTION_VIZ` | `True` | Show attention map overlay on answers |
| `CHAIN_OF_THOUGHT` | `False` | Enable reasoning chain mode |

> Copy `.env.example` to `.env` and populate required values before running.

---

## Project Structure

```
ImageReasoning/
├── README.md
├── requirements.txt
├── ImAgE.py
└── ...
```

---

## Roadmap

- [ ] Video VQA: temporal question answering across video frames (what happens after the person sits down?)
- [ ] Medical imaging VQA: fine-tuned model for radiology image Q&A (not for clinical use)
- [ ] Multilingual VQA: answer questions posed in any language about any image
- [ ] Active grounding: draw bounding box around the object referenced in the answer
- [ ] Comparative benchmarking dashboard: automated weekly comparison of new VLM releases on standard benchmarks

---

## Contributing

Contributions, issues, and suggestions are welcome.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-idea`
3. Commit your changes: `git commit -m 'feat: add your idea'`
4. Push to your branch: `git push origin feature/your-idea`
5. Open a Pull Request with a clear description

Please follow conventional commit messages and add documentation for new features.

---

## Notes

BLIP-2 model weights require ~5GB of disk space and benefit significantly from GPU inference (A100 or equivalent for full model; T4 or equivalent for the smaller 2.7B variant). CPU inference is functional for testing but slow (~30–60s per query). GPT-4 Vision requires an OpenAI API key and incurs per-image API costs.

---

## Author

**Devanik Debnath**  
B.Tech, Electronics & Communication Engineering  
National Institute of Technology Agartala

[![GitHub](https://img.shields.io/badge/GitHub-Devanik21-black?style=flat-square&logo=github)](https://github.com/Devanik21)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-devanik-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/devanik/)

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

*Built with curiosity, depth, and care — because good projects deserve good documentation.*
