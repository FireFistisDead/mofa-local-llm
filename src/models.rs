use serde::{Deserialize, Serialize};

/// All supported model categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelCategory {
    LLM,
    VLM,
    ASR,
    TTS,
    ImageGen,
}

impl std::fmt::Display for ModelCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelCategory::LLM => write!(f, "LLM"),
            ModelCategory::VLM => write!(f, "VLM"),
            ModelCategory::ASR => write!(f, "ASR"),
            ModelCategory::TTS => write!(f, "TTS"),
            ModelCategory::ImageGen => write!(f, "Image Generation"),
        }
    }
}

/// A known model definition from the OminiX-MLX catalog.
#[derive(Debug, Clone)]
pub struct ModelDef {
    pub id: &'static str,
    pub name: &'static str,
    pub category: ModelCategory,
    pub engine: &'static str,
    pub repo_id: &'static str,
    pub description: &'static str,
    pub size_hint: &'static str,
}

/// All models mentioned in OminiX-MLX README.
pub static MODEL_CATALOG: &[ModelDef] = &[
    // LLMs
    ModelDef {
        id: "qwen2.5-0.5b-instruct",
        name: "Qwen2.5-0.5B Instruct",
        category: ModelCategory::LLM,
        engine: "qwen2",
        repo_id: "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        description: "Smallest Qwen2.5, great for testing",
        size_hint: "~400MB",
    },
    ModelDef {
        id: "qwen3-4b-bf16",
        name: "Qwen3-4B",
        category: ModelCategory::LLM,
        engine: "qwen3",
        repo_id: "mlx-community/Qwen3-4B-bf16",
        description: "Qwen3 4B parameters, bf16",
        size_hint: "~8GB",
    },
    ModelDef {
        id: "mistral-7b-instruct-4bit",
        name: "Mistral-7B Instruct 4-bit",
        category: ModelCategory::LLM,
        engine: "mistral",
        repo_id: "mlx-community/Mistral-7B-Instruct-v0.2-4bit",
        description: "Mistral 7B, 4-bit quantized, ~74 tok/s",
        size_hint: "~4GB",
    },
    ModelDef {
        id: "glm4-9b-chat-4bit",
        name: "GLM-4-9B Chat 4-bit",
        category: ModelCategory::LLM,
        engine: "glm4",
        repo_id: "mlx-community/glm-4-9b-chat-4bit",
        description: "GLM-4 9B with partial RoPE",
        size_hint: "~5GB",
    },
    ModelDef {
        id: "mixtral-8x7b-instruct-4bit",
        name: "Mixtral-8x7B Instruct 4-bit",
        category: ModelCategory::LLM,
        engine: "mixtral",
        repo_id: "mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit",
        description: "Mixtral MoE 8x7B, 4-bit quantized",
        size_hint: "~26GB",
    },
    ModelDef {
        id: "minicpm-sala-9b-8bit",
        name: "MiniCPM-SALA 9B 8-bit",
        category: ModelCategory::LLM,
        engine: "minicpm-sala",
        repo_id: "moxin-org/MiniCPM4-SALA-9B-8bit-mlx",
        description: "Hybrid attention, 1M context window",
        size_hint: "~9.6GB",
    },
    // VLMs
    ModelDef {
        id: "moxin-7b-vlm-8bit",
        name: "Moxin-7B VLM 8-bit",
        category: ModelCategory::VLM,
        engine: "moxin-vlm",
        repo_id: "moxin-org/Moxin-7B-VLM-8bit-mlx",
        description: "DINOv2 + SigLIP + Mistral-7B, 30 tok/s",
        size_hint: "~10GB",
    },
    // ASR
    ModelDef {
        id: "funasr-paraformer",
        name: "FunASR Paraformer-large",
        category: ModelCategory::ASR,
        engine: "funasr",
        repo_id: "OminiX-ai/paraformer-mlx",
        description: "Non-autoregressive Chinese/English ASR, 18x real-time",
        size_hint: "~500MB",
    },
    // TTS
    ModelDef {
        id: "gpt-sovits",
        name: "GPT-SoVITS",
        category: ModelCategory::TTS,
        engine: "gpt-sovits",
        repo_id: "OminiX-ai/gpt-sovits-mlx",
        description: "Few-shot voice cloning, 4x real-time",
        size_hint: "~2GB",
    },
    // Image Generation
    ModelDef {
        id: "zimage-turbo",
        name: "Z-Image-Turbo",
        category: ModelCategory::ImageGen,
        engine: "zimage",
        repo_id: "uqer1244/MLX-z-image",
        description: "Fast image generation, 6B S3-DiT",
        size_hint: "~8GB",
    },
    ModelDef {
        id: "flux-klein-4b",
        name: "FLUX.2-klein-4B",
        category: ModelCategory::ImageGen,
        engine: "flux-klein",
        repo_id: "black-forest-labs/FLUX.2-klein-4B",
        description: "FLUX.2 image generation with Qwen3 encoder",
        size_hint: "~13GB",
    },
];

impl ModelDef {
    pub fn find_by_id(id: &str) -> Option<&'static ModelDef> {
        MODEL_CATALOG.iter().find(|m| m.id == id)
    }

    pub fn find_by_engine(engine: &str) -> Vec<&'static ModelDef> {
        MODEL_CATALOG
            .iter()
            .filter(|m| m.engine == engine)
            .collect()
    }

    pub fn by_category(category: ModelCategory) -> Vec<&'static ModelDef> {
        MODEL_CATALOG
            .iter()
            .filter(|m| m.category == category)
            .collect()
    }
}
