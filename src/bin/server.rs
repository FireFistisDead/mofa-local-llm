use std::path::PathBuf;
use std::time::Instant;

use mofa_local_llm::config::AppConfig;
use mofa_local_llm::inference::llm::{LlmBackend, LlmEngine};
use mofa_local_llm::inference::InferenceRequest;

fn detect_backend(model_type: &str) -> Option<LlmBackend> {
    match model_type {
        "qwen2" => Some(LlmBackend::Qwen2),
        "qwen3" | "qwen" => Some(LlmBackend::Qwen3),
        "mistral" => Some(LlmBackend::Mistral),
        "glm4" | "chatglm" => Some(LlmBackend::Glm4),
        "mixtral" => Some(LlmBackend::Mixtral),
        "minicpm" | "minicpm4" => Some(LlmBackend::MiniCpmSala),
        _ => None,
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let args: Vec<String> = std::env::args().collect();

    let port: u16 = args
        .iter()
        .position(|a| a == "--port")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(8080);

    let model_path = args
        .iter()
        .position(|a| a == "--model")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.to_string());

    let models_dir_override = args
        .iter()
        .position(|a| a == "--models-dir")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str());

    // Load config
    let mut config = AppConfig::load(models_dir_override);
    config.scan_models_dir();
    let _ = config.save();

    // Inference channel
    let (inference_tx, mut inference_rx) =
        tokio::sync::mpsc::channel::<InferenceRequest>(4);

    let loaded_model_id: Option<String>;

    // Load model if specified
    if let Some(ref model_path) = model_path {
        let path = PathBuf::from(model_path);
        let model_id = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        // Detect model type
        let config_json = path.join("config.json");
        let model_type = if config_json.exists() {
            let content = std::fs::read_to_string(&config_json)?;
            serde_json::from_str::<serde_json::Value>(&content)
                .ok()
                .and_then(|v| v.get("model_type").and_then(|v| v.as_str()).map(|s| s.to_string()))
                .unwrap_or_else(|| "qwen2".to_string())
        } else {
            "qwen2".to_string()
        };

        let backend = detect_backend(&model_type)
            .unwrap_or(LlmBackend::Qwen3);

        eprintln!("Loading model: {} (type: {}, backend: {:?})", model_id, model_type, backend);
        let t0 = Instant::now();

        let mut llm = LlmEngine::load(&path, backend, &model_id)
            .map_err(|e| anyhow::anyhow!(e))?;
        eprintln!("Model loaded in {:.1}s", t0.elapsed().as_secs_f64());

        loaded_model_id = Some(model_id);

        // Spawn inference worker
        std::thread::spawn(move || {
            while let Some(req) = inference_rx.blocking_recv() {
                match req {
                    InferenceRequest::Chat {
                        messages,
                        max_tokens,
                        temperature,
                        response_tx,
                    } => {
                        let result = llm.chat(&messages, max_tokens, temperature);
                        let _ = response_tx.send(result);
                    }
                    InferenceRequest::Shutdown => break,
                    _ => {
                        // Other request types not handled by LLM worker
                    }
                }
            }
        });
    } else {
        loaded_model_id = None;
        eprintln!("No model specified. Use --model <path> to load a model.");
        eprintln!("Server will start in download-only mode.");

        // Spawn a dummy worker that rejects inference requests
        std::thread::spawn(move || {
            while let Some(req) = inference_rx.blocking_recv() {
                match req {
                    InferenceRequest::Chat { response_tx, .. } => {
                        let _ = response_tx.send(Err(
                            "No model loaded. Use POST /v1/models/download to download one."
                                .to_string(),
                        ));
                    }
                    InferenceRequest::Transcribe { response_tx, .. } => {
                        let _ = response_tx.send(Err("No ASR model loaded.".to_string()));
                    }
                    InferenceRequest::Synthesize { response_tx, .. } => {
                        let _ = response_tx.send(Err("No TTS model loaded.".to_string()));
                    }
                    InferenceRequest::GenerateImage { response_tx, .. } => {
                        let _ = response_tx.send(Err("No image model loaded.".to_string()));
                    }
                    InferenceRequest::Shutdown => break,
                }
            }
        });
    }

    // Start server
    mofa_local_llm::api::server::start_server(
        port,
        inference_tx,
        config,
        loaded_model_id,
    )
    .await?;

    Ok(())
}
