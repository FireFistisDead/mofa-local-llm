use axum::extract::State;
use axum::http::StatusCode;
use axum::response::Json;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Arc;

use super::types::*;
use crate::config::AppConfig;
use crate::download;
use crate::inference::{ChatResult, InferenceRequest};

use tokio::sync::{oneshot, RwLock};

/// Shared server state.
pub struct AppState {
    pub inference_tx: tokio::sync::mpsc::Sender<InferenceRequest>,
    pub config: RwLock<AppConfig>,
    pub default_temperature: f32,
    pub default_max_tokens: usize,
    pub loaded_model_id: RwLock<Option<String>>,
}

pub type SharedState = Arc<AppState>;

// ============================================================================
// Health Check
// ============================================================================

pub async fn health_check() -> Json<Value> {
    Json(json!({"status": "ok"}))
}

// ============================================================================
// Chat Completion
// ============================================================================

pub async fn chat_completion(
    State(state): State<SharedState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, (StatusCode, Json<ApiError>)> {
    if req.messages.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ApiError::new("messages array is empty", "invalid_request_error")),
        ));
    }

    let messages: Vec<(String, String)> = req
        .messages
        .iter()
        .map(|m| (m.role.clone(), m.content.clone()))
        .collect();

    let max_tokens = req.max_tokens.unwrap_or(state.default_max_tokens);
    let temperature = req.temperature.unwrap_or(state.default_temperature);

    let (resp_tx, resp_rx) = oneshot::channel();

    state
        .inference_tx
        .send(InferenceRequest::Chat {
            messages,
            max_tokens,
            temperature,
            response_tx: resp_tx,
        })
        .await
        .map_err(|_| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiError::new("Inference worker unavailable", "server_error")),
            )
        })?;

    let result = resp_rx.await.map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiError::new("Inference channel closed", "server_error")),
        )
    })?;

    match result {
        Ok(chat_result) => {
            eprintln!(
                "[api] Generated {} tokens ({:.0}ms prefill, {:.1} tok/s)",
                chat_result.completion_tokens, chat_result.prefill_ms, chat_result.decode_tps
            );

            Ok(Json(ChatCompletionResponse {
                id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
                object: "chat.completion".to_string(),
                model: req.model,
                choices: vec![Choice {
                    index: 0,
                    message: ResponseMessage {
                        role: "assistant".to_string(),
                        content: chat_result.text,
                    },
                    finish_reason: "stop".to_string(),
                }],
                usage: Usage {
                    prompt_tokens: chat_result.prompt_tokens,
                    completion_tokens: chat_result.completion_tokens,
                    total_tokens: chat_result.prompt_tokens + chat_result.completion_tokens,
                },
            }))
        }
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiError::new(e, "server_error")),
        )),
    }
}

// ============================================================================
// Model Management
// ============================================================================

pub async fn list_models(
    State(state): State<SharedState>,
) -> Json<ModelListResponse> {
    let mut config = state.config.write().await;
    config.scan_models_dir();

    let loaded_id = state.loaded_model_id.read().await;

    let data: Vec<ModelInfo> = config
        .models
        .iter()
        .map(|m| {
            let is_loaded = loaded_id.as_deref() == Some(&m.id);
            ModelInfo {
                id: m.id.clone(),
                object: "model".to_string(),
                repo_id: if m.repo_id.is_empty() {
                    None
                } else {
                    Some(m.repo_id.clone())
                },
                model_type: m.model_type.clone(),
                path: m.path.clone(),
                loaded: is_loaded,
                size_bytes: m.size_bytes,
                downloaded_at: m.downloaded_at.clone(),
            }
        })
        .collect();

    Json(ModelListResponse {
        object: "list".to_string(),
        data,
    })
}

pub async fn download_model(
    State(state): State<SharedState>,
    Json(req): Json<DownloadRequest>,
) -> Result<(StatusCode, Json<Value>), (StatusCode, Json<ApiError>)> {
    if req.repo_id.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ApiError::new("repo_id is required", "invalid_request_error")),
        ));
    }

    let model_id = req
        .repo_id
        .split('/')
        .last()
        .unwrap_or(&req.repo_id)
        .to_string();

    // Check if already exists
    {
        let config = state.config.read().await;
        if config.models.iter().any(|m| m.id == model_id) {
            return Err((
                StatusCode::CONFLICT,
                Json(ApiError::new(
                    format!("Model '{}' already exists", model_id),
                    "conflict",
                )),
            ));
        }
    }

    let models_dir = PathBuf::from(&state.config.read().await.models_dir);
    let repo_id = req.repo_id.clone();
    let state_clone = state.clone();

    tokio::task::spawn_blocking(move || {
        eprintln!("[download] Starting: {}", repo_id);
        match download::download_model(&repo_id, &models_dir, None) {
            Ok(entry) => {
                let mut config = state_clone.config.blocking_write();
                download::register_model(&mut config, entry);
                eprintln!("[download] Complete: {}", repo_id);
            }
            Err(e) => {
                eprintln!("[download] Failed: {}: {}", repo_id, e);
                let dest = models_dir.join(repo_id.split('/').last().unwrap_or(&repo_id));
                let _ = std::fs::remove_dir_all(&dest);
            }
        }
    });

    Ok((
        StatusCode::ACCEPTED,
        Json(json!({
            "status": "downloading",
            "id": model_id,
            "repo_id": req.repo_id,
        })),
    ))
}

pub async fn delete_model(
    State(state): State<SharedState>,
    axum::extract::Path(model_id): axum::extract::Path<String>,
) -> Result<Json<Value>, (StatusCode, Json<ApiError>)> {
    let mut config = state.config.write().await;

    let idx = config
        .models
        .iter()
        .position(|m| m.id == model_id)
        .ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                Json(ApiError::new(
                    format!("Model not found: {}", model_id),
                    "not_found",
                )),
            )
        })?;

    // Prevent deleting loaded model
    let loaded_id = state.loaded_model_id.read().await;
    if loaded_id.as_deref() == Some(&model_id) {
        return Err((
            StatusCode::CONFLICT,
            Json(ApiError::new(
                "Cannot delete the currently loaded model",
                "conflict",
            )),
        ));
    }

    let path = PathBuf::from(&config.models[idx].path);
    if path.exists() {
        std::fs::remove_dir_all(&path).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiError::new(
                    format!("Failed to remove directory: {}", e),
                    "server_error",
                )),
            )
        })?;
    }

    config.models.remove(idx);
    let _ = config.save();

    Ok(Json(json!({"id": model_id, "deleted": true})))
}

// ============================================================================
// Audio Transcription
// ============================================================================

pub async fn transcribe_audio(
    State(state): State<SharedState>,
    body: axum::body::Bytes,
) -> Result<Json<TranscriptionResponse>, (StatusCode, Json<ApiError>)> {
    let (resp_tx, resp_rx) = oneshot::channel();

    // Parse WAV from body bytes
    let samples = parse_wav_bytes(&body).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(ApiError::new(e, "invalid_request_error")),
        )
    })?;

    state
        .inference_tx
        .send(InferenceRequest::Transcribe {
            audio_samples: samples,
            sample_rate: 16000,
            response_tx: resp_tx,
        })
        .await
        .map_err(|_| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiError::new("Inference worker unavailable", "server_error")),
            )
        })?;

    let result = resp_rx.await.map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiError::new("Inference channel closed", "server_error")),
        )
    })?;

    match result {
        Ok(text) => Ok(Json(TranscriptionResponse { text })),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiError::new(e, "server_error")),
        )),
    }
}

fn parse_wav_bytes(bytes: &[u8]) -> Result<Vec<f32>, String> {
    let cursor = std::io::Cursor::new(bytes);
    let reader = hound::WavReader::new(cursor).map_err(|e| format!("Invalid WAV: {}", e))?;
    let spec = reader.spec();

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .filter_map(|s| s.ok())
            .collect(),
        hound::SampleFormat::Int => {
            let max_val = (1 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max_val)
                .collect()
        }
    };

    // Convert stereo to mono if needed
    let mono = if spec.channels == 2 {
        samples
            .chunks(2)
            .map(|c| (c[0] + c.get(1).copied().unwrap_or(0.0)) / 2.0)
            .collect()
    } else {
        samples
    };

    // Resample to 16kHz if needed
    Ok(crate::inference::asr::resample_to_16khz(&mono, spec.sample_rate))
}

// ============================================================================
// Model Catalog
// ============================================================================

pub async fn list_catalog() -> Json<Value> {
    let catalog: Vec<Value> = crate::models::MODEL_CATALOG
        .iter()
        .map(|m| {
            json!({
                "id": m.id,
                "name": m.name,
                "category": format!("{}", m.category),
                "engine": m.engine,
                "repo_id": m.repo_id,
                "description": m.description,
                "size_hint": m.size_hint,
            })
        })
        .collect();

    Json(json!({"object": "list", "data": catalog}))
}
