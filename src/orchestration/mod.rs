use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::path::PathBuf;

use crate::inference::llm::LlmEngine;
use crate::inference::asr::AsrEngine;
use crate::inference::tts::TtsEngine;
use crate::inference::image_gen::ImageGenEngine;
use crate::config::AppConfig;

/// Types of models supported by the orchestrator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    LLM,
    VLM,
    ASR,
    TTS,
    ImageGen,
}

/// Trait for any model managed by the pool.
pub trait ManagedModel: Send + Sync {
    fn id(&self) -> &str;
    fn model_type(&self) -> ModelType;
    fn as_any(&self) -> &dyn std::any::Any;
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

// Wrappers to implement ManagedModel for specific engines
// We need wrappers because the engines themselves might not implement Send+Sync or we want to add metadata.
// For the GSoC prototype, we assume the MLX engines are Send. (They are usually valid to send across threads).

pub struct ManagedLlm(pub LlmEngine);
impl ManagedModel for ManagedLlm {
    fn id(&self) -> &str { self.0.model_id() }
    fn model_type(&self) -> ModelType { ModelType::LLM }
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

pub struct ManagedAsr(pub AsrEngine);
impl ManagedModel for ManagedAsr {
    fn id(&self) -> &str { self.0.model_id() }
    fn model_type(&self) -> ModelType { ModelType::ASR }
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

pub struct ManagedTts(pub TtsEngine);
impl ManagedModel for ManagedTts {
    fn id(&self) -> &str { self.0.model_id() }
    fn model_type(&self) -> ModelType { ModelType::TTS }
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

/// Manages a pool of loaded models.
pub struct ModelPool {
    models: HashMap<String, Box<dyn ManagedModel>>,
    config: Arc<Mutex<AppConfig>>,
}

impl ModelPool {
    pub fn new(config: AppConfig) -> Self {
        Self {
            models: HashMap::new(),
            config: Arc::new(Mutex::new(config)),
        }
    }

    pub fn get_model(&mut self, model_id: &str) -> Option<&mut dyn ManagedModel> {
        self.models.get_mut(model_id).map(|b| b.as_mut())
    }
    
    pub fn is_loaded(&self, model_id: &str) -> bool {
        self.models.contains_key(model_id)
    }

    pub fn list_loaded(&self) -> Vec<String> {
        self.models.keys().cloned().collect()
    }
    
    pub fn load_model(&mut self, model_id: &str) -> Result<(), String> {
        if self.models.contains_key(model_id) {
            return Ok(());
        }

        let config = self.config.lock().unwrap();
        let entry = config.models.iter().find(|m| m.id == model_id)
            .ok_or_else(|| format!("Model '{}' not found in configuration", model_id))?
            .clone();
        drop(config); // Unlock early
        
        let path = PathBuf::from(&entry.path);
        
        // Attempt to detect LLM backend
        if let Some(backend) = crate::inference::llm::LlmBackend::from_model_type(&entry.model_type) {
            let engine = LlmEngine::load(&path, backend, model_id)?;
            self.models.insert(model_id.to_string(), Box::new(ManagedLlm(engine)));
            return Ok(());
        }
        
        // Other types
        match entry.model_type.as_str() {
            "funasr" | "paraformer" => {
                 let engine = AsrEngine::load(&path, model_id)?;
                 self.models.insert(model_id.to_string(), Box::new(ManagedAsr(engine)));
                 Ok(())
            },
            "gpt-sovits" => {
                 let engine = TtsEngine::load(&path, model_id)?;
                 self.models.insert(model_id.to_string(), Box::new(ManagedTts(engine)));
                 Ok(())
            },
            _ => Err(format!("Unsupported model type: {} for model {}", entry.model_type, model_id)),
        }
    }
