use serde::{Deserialize, Serialize};

// ============================================================================
// Chat Completion (OpenAI-compatible)
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default = "default_model")]
    pub model: String,
    #[serde(default)]
    pub stream: bool,
}

fn default_model() -> String {
    "default".to_string()
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct Choice {
    pub index: usize,
    pub message: ResponseMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct ResponseMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

// ============================================================================
// Model Management
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct DownloadRequest {
    pub repo_id: String,
    #[serde(default)]
    pub model_type: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct LoadModelRequest {
    pub model_id: String,
    #[serde(default)]
    pub engine: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repo_id: Option<String>,
    pub model_type: String,
    pub path: String,
    pub loaded: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub downloaded_at: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ModelListResponse {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

// ============================================================================
// Audio (ASR / TTS)
// ============================================================================

#[derive(Debug, Serialize)]
pub struct TranscriptionResponse {
    pub text: String,
}

// ============================================================================
// Image Generation
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct ImageGenerationRequest {
    pub prompt: String,
    #[serde(default = "default_size")]
    pub size: String,
    #[serde(default = "default_steps")]
    pub n_steps: u32,
}

fn default_size() -> String {
    "512x512".to_string()
}

fn default_steps() -> u32 {
    4
}

#[derive(Debug, Serialize)]
pub struct ImageGenerationResponse {
    pub data: Vec<ImageData>,
}

#[derive(Debug, Serialize)]
pub struct ImageData {
    pub b64_json: String,
}

// ============================================================================
// Error
// ============================================================================

#[derive(Debug, Serialize)]
pub struct ApiError {
    pub error: ApiErrorDetail,
}

#[derive(Debug, Serialize)]
pub struct ApiErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
}

impl ApiError {
    pub fn new(message: impl Into<String>, error_type: impl Into<String>) -> Self {
        ApiError {
            error: ApiErrorDetail {
                message: message.into(),
                error_type: error_type.into(),
            },
        }
    }
}
