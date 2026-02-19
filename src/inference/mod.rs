pub mod llm;
pub mod vlm;
pub mod asr;
pub mod tts;
pub mod image_gen;

use std::path::Path;
use std::sync::mpsc;

/// Common inference request routed to an inference worker thread.
pub enum InferenceRequest {
    /// Chat completion (LLM or VLM)
    Chat {
        model_id: String,
        messages: Vec<(String, String)>,
        max_tokens: usize,
        temperature: f32,
        response_tx: tokio::sync::oneshot::Sender<Result<ChatResult, String>>,
    },
    /// Speech-to-text (ASR)
    Transcribe {
        model_id: String,
        audio_samples: Vec<f32>,
        sample_rate: u32,
        response_tx: tokio::sync::oneshot::Sender<Result<String, String>>,
    },
    /// Text-to-speech (TTS)
    Synthesize {
        model_id: String,
        text: String,
        reference_audio: Option<String>,
        response_tx: tokio::sync::oneshot::Sender<Result<Vec<f32>, String>>,
    },
    /// Image generation
    GenerateImage {
        model_id: String,
        prompt: String,
        width: u32,
        height: u32,
        steps: u32,
        response_tx: tokio::sync::oneshot::Sender<Result<ImageResult, String>>,
    },
    /// Shutdown
    Shutdown,
}

#[derive(Debug, Clone)]
pub struct ChatResult {
    pub text: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub prefill_ms: f64,
    pub decode_tps: f64,
}

#[derive(Debug, Clone)]
pub struct ImageResult {
    pub pixels: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

/// Trait for all inference engines.
pub trait InferenceEngine: Send {
    fn engine_name(&self) -> &str;
    fn model_id(&self) -> &str;
}
