# MOFA Local LLM - Task Document

## Overview

A complete local ML inference platform for Apple Silicon, powered by **OminiX-MLX** (pure Rust + Metal GPU). Provides both an OpenAI-compatible HTTP API server and a native GUI application.

**Supported model types:**
- LLM: Qwen2, Qwen3, Mistral, GLM-4, Mixtral, MiniCPM-SALA
- VLM: Moxin-7B (DINOv2 + SigLIP + Mistral-7B)
- ASR: FunASR Paraformer (Chinese/English)
- TTS: GPT-SoVITS (voice cloning)
- Image Generation: FLUX.2-klein, Z-Image

## Architecture

```
mofa-local-llm/
├── src/
│   ├── lib.rs              # Library root
│   ├── config.rs           # AppConfig: model directory, scan, persist
│   ├── models.rs           # MODEL_CATALOG: all known model definitions
│   ├── download.rs         # HuggingFace model downloading (hf-hub)
│   ├── inference/
│   │   ├── mod.rs          # InferenceRequest enum, ChatResult, ImageResult
│   │   ├── llm.rs          # LlmEngine: Qwen2/3, Mistral, GLM4, Mixtral, MiniCPM-SALA
│   │   ├── vlm.rs          # VlmEngine: Moxin-7B with image understanding
│   │   ├── asr.rs          # AsrEngine: FunASR Paraformer transcription
│   │   ├── tts.rs          # TtsEngine: GPT-SoVITS voice synthesis
│   │   └── image_gen.rs    # ImageGenEngine: FLUX.2-klein, Z-Image
│   ├── api/
│   │   ├── mod.rs
│   │   ├── types.rs        # OpenAI-compatible request/response types
│   │   ├── handlers.rs     # Route handlers (chat, models, download, etc.)
│   │   └── server.rs       # Axum router + CORS + server startup
│   ├── gui/
│   │   ├── mod.rs
│   │   └── app.rs          # egui/eframe desktop application
│   └── bin/
│       ├── server.rs       # CLI server: --model <path> --port <port>
│       └── gui.rs          # GUI application entry point
├── Cargo.toml              # Dependencies (OminiX-MLX crates as path deps)
└── TASK.md                 # This file
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | OpenAI-compatible chat completion |
| GET | `/v1/models` | List downloaded models |
| POST | `/v1/models/download` | Download model from HuggingFace |
| DELETE | `/v1/models/{id}` | Delete a downloaded model |
| POST | `/v1/audio/transcriptions` | Transcribe audio (WAV) |
| GET | `/v1/catalog` | List all available models in catalog |
| GET | `/health` | Health check |

## Usage

### Server

```bash
# Build
cargo build --release --bin mofa-server

# Run with a model
./target/release/mofa-server --model ~/.mofa/models/Qwen2.5-0.5B-Instruct-4bit --port 8090

# Test
curl http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-0.5B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 128,
    "temperature": 0.7
  }'

# Download a model via API
curl -X POST http://localhost:8090/v1/models/download \
  -H "Content-Type: application/json" \
  -d '{"catalog_id": "qwen2.5-0.5b-instruct"}'
```

### GUI

```bash
cargo build --release --bin mofa-gui
./target/release/mofa-gui
```

## Model Catalog

| ID | Name | Category | HuggingFace Repo | Size |
|----|------|----------|-----------------|------|
| qwen2.5-0.5b-instruct | Qwen2.5-0.5B Instruct | LLM | mlx-community/Qwen2.5-0.5B-Instruct-4bit | ~400MB |
| qwen3-4b-bf16 | Qwen3-4B | LLM | mlx-community/Qwen3-4B-bf16 | ~8GB |
| mistral-7b-instruct-4bit | Mistral-7B Instruct | LLM | mlx-community/Mistral-7B-Instruct-v0.2-4bit | ~4GB |
| glm4-9b-chat-4bit | GLM-4-9B Chat | LLM | mlx-community/glm-4-9b-chat-4bit | ~5GB |
| mixtral-8x7b-instruct-4bit | Mixtral-8x7B Instruct | LLM | mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit | ~26GB |
| minicpm-sala-9b-8bit | MiniCPM-SALA 9B | LLM | moxin-org/MiniCPM4-SALA-9B-8bit-mlx | ~9.6GB |
| moxin-7b-vlm-8bit | Moxin-7B VLM | VLM | moxin-org/Moxin-7B-VLM-8bit-mlx | ~10GB |
| funasr-paraformer | FunASR Paraformer | ASR | OminiX-ai/paraformer-mlx | ~500MB |
| gpt-sovits | GPT-SoVITS | TTS | OminiX-ai/gpt-sovits-mlx | ~2GB |
| zimage-turbo | Z-Image-Turbo | Image Gen | uqer1244/MLX-z-image | ~8GB |
| flux-klein-4b | FLUX.2-klein-4B | Image Gen | black-forest-labs/FLUX.2-klein-4B | ~13GB |

Models are stored in `~/.mofa/models/`. Configuration is at `~/.mofa/ominix-config.json`.

## Key Design Decisions

1. **Inference threading**: A dedicated OS thread handles all inference work, communicating with the async HTTP server via tokio mpsc channels. This avoids blocking the async runtime with CPU/GPU-bound work.

2. **Backend detection**: The server reads `config.json` from the model directory and uses the `model_type` field to select the appropriate backend (qwen2, qwen3, mistral, glm4, mixtral, minicpm).

3. **ChatML prompt format**: All LLM backends (except MiniCPM-SALA) use the ChatML `<|im_start|>/<|im_end|>` prompt format.

4. **Quantized model support**: The Qwen2 backend supports 4-bit quantized models with group_size=64 via manual weight loading (including attention biases).

## Bug Fixes Applied to OminiX-MLX

### qwen3-mlx/src/qwen2.rs - Quantized linear bias loading

The `make_quantized_linear` function was not loading the linear bias (`{prefix}.bias`) for attention projections. Qwen2 models use `bias=True` for q/k/v projections, but the quantized loader only loaded quantization biases (`{prefix}.biases`), resulting in garbage output. Fixed by also loading `{prefix}.bias` when present.

## Dependencies

All OminiX-MLX crates are linked as path dependencies from `../OminiX-MLX/`:
- mlx-rs, mlx-rs-core (MLX Rust bindings)
- qwen3-mlx (Qwen2/3 models)
- glm4-mlx, mistral-mlx, mixtral-mlx (other LLMs)
- minicpm-sala-mlx (MiniCPM-SALA with 1M context)
- moxin-vlm-mlx (vision-language model)
- funasr-mlx, funasr-nano-mlx (speech recognition)
- gpt-sovits-mlx (text-to-speech)
- flux-klein-mlx, zimage-mlx (image generation)

External dependencies: axum, tokio, serde, eframe/egui, hf-hub, tokenizers, image, cpal, hound.

## Testing Status

| Component | Status | Notes |
|-----------|--------|-------|
| Server build | Pass | Release binary compiles |
| Qwen2 LLM inference | Pass | Tested with Qwen2.5-0.5B-Instruct-4bit |
| Chat API | Pass | English and Chinese responses verified |
| Health check | Pass | |
| Model listing | Pass | |
| GUI build | Pass | Release binary compiles |
| Qwen3 inference | Untested | Needs Qwen3 model download |
| Mistral inference | Untested | Needs model download |
| GLM-4 inference | Untested | Needs model download |
| Mixtral inference | Untested | Needs model download |
| MiniCPM-SALA inference | Untested | Needs model download |
| VLM inference | Untested | Needs model download |
| ASR inference | Untested | Needs model download |
| TTS inference | Untested | Needs model download |
| Image generation | Untested | Needs model download |
