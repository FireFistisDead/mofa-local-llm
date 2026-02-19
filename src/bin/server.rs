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

    let models_dir_override = args
        .iter()
        .position(|a| a == "--models-dir")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str());

    // Load config
    let mut config = AppConfig::load(models_dir_override);
    config.scan_models_dir();
    let _ = config.save();
    
    // Initialize Model Pool
    let mut model_pool = mofa_local_llm::orchestration::ModelPool::new(config.clone());

    // Inference channel
    let (inference_tx, mut inference_rx) =
        tokio::sync::mpsc::channel::<InferenceRequest>(16);

    // Spawn inference orchestrator
    std::thread::spawn(move || {
        while let Some(req) = inference_rx.blocking_recv() {
            // Helper to handle loading model if needed
            let ensure_model = |pool: &mut mofa_local_llm::orchestration::ModelPool, model_id: &str| -> Result<(), String> {
                if !pool.is_loaded(model_id) {
                    eprintln!("[orchestrator] Model '{}' not loaded. Loading now...", model_id);
                    if let Err(e) = pool.load_model(model_id) {
                        return Err(format!("Failed to load model '{}': {}", model_id, e));
                    }
                }
                Ok(())
            };

            match req {
                InferenceRequest::Chat {
                    model_id,
                    messages,
                    max_tokens,
                    temperature,
                    response_tx,
                } => {
                    let res = (|| {
                        ensure_model(&mut model_pool, &model_id)?;
                        let model = model_pool.get_model(&model_id)
                             .ok_or("Model not found in pool after load attempt")?;
                        
                         if let Some(llm) = model.as_any_mut().downcast_mut::<mofa_local_llm::orchestration::ManagedLlm>() {
                             llm.0.chat(&messages, max_tokens, temperature)
                         } else {
                             Err(format!("Model '{}' is not an LLM", model_id))
                         }
                    })();
                    let _ = response_tx.send(res);
                }
                InferenceRequest::Transcribe {
                    model_id,
                    audio_samples,
                    sample_rate,
                    response_tx,
                } => {
                    let res = (|| {
                        ensure_model(&mut model_pool, &model_id)?;
                        let model = model_pool.get_model(&model_id)
                             .ok_or("Model not found in pool after load attempt")?;
                             
                        if let Some(asr) = model.as_any_mut().downcast_mut::<mofa_local_llm::orchestration::ManagedAsr>() {
                             // Resample if needed
                             let samples = if sample_rate != 16000 {
                                 mofa_local_llm::inference::asr::resample_to_16khz(&audio_samples, sample_rate)
                             } else {
                                 audio_samples
                             };
                             asr.0.transcribe(&samples)
                        } else {
                             Err(format!("Model '{}' is not an ASR model", model_id))
                        }
                    })();
                    let _ = response_tx.send(res);
                }
                InferenceRequest::Synthesize {
                     model_id,
                     text,
                     reference_audio,
                     response_tx
                } => {
                     let res = (|| {
                        ensure_model(&mut model_pool, &model_id)?;
                        let model = model_pool.get_model(&model_id)
                             .ok_or("Model not found")?;

                        if let Some(tts) = model.as_any_mut().downcast_mut::<mofa_local_llm::orchestration::ManagedTts>() {
                             if let Some(ref_audio) = reference_audio {
                                 tts.0.set_reference(&ref_audio)?;
                             }
                             tts.0.synthesize(&text)
                        } else {
                             Err(format!("Model '{}' is not a TTS model", model_id))
                        }
                     })();
                     let _ = response_tx.send(res);
                }
                InferenceRequest::GenerateImage {
                     model_id,
                     response_tx,
                     ..
                } => {
                     let _ = response_tx.send(Err(format!("Image generation not yet implemented in orchestrator for model {}", model_id)));
                }
                InferenceRequest::Shutdown => break,
            }
        }
    });

    // Start server
    // We pass None as loaded_model_id because now the pool manages it dynamically
    eprintln!("Starting Multi-Model Orchestrator Server on port {}", port);
    mofa_local_llm::api::server::start_server(
        port,
        inference_tx,
        config,
        None, // No single loaded model ID
    )
    .await?;

    Ok(())
}
