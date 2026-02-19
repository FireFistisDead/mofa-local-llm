use std::path::Path;
use std::time::Instant;

use mlx_rs::ops::indexing::{IndexOp, NewAxis};
use mlx_rs::transforms::eval;
use mlx_rs::Array;

use super::ChatResult;

/// Supported LLM engine backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmBackend {
    Qwen2,
    Qwen3,
    Mistral,
    Glm4,
    Mixtral,
    MiniCpmSala,
}

impl LlmBackend {
    pub fn from_model_type(model_type: &str) -> Option<Self> {
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
}


/// A loaded LLM model with its tokenizer.
pub struct LlmEngine {
    backend: LlmBackend,
    model_id: String,
    inner: LlmInner,
    tokenizer: tokenizers::Tokenizer,
}

enum LlmInner {
    Qwen2(qwen3_mlx::qwen2::Model),
    Qwen3(qwen3_mlx::Model),
    Mistral(mistral_mlx::Model),
    Glm4(glm4_mlx::Model),
    Mixtral(mixtral_mlx::Model),
    MiniCpmSala(minicpm_sala_mlx::Model),
}

impl LlmEngine {
    /// Load an LLM model from a directory.
    pub fn load(model_dir: &Path, backend: LlmBackend, model_id: &str) -> Result<Self, String> {
        let model_dir_str = model_dir.to_string_lossy();
        eprintln!("[llm] Loading {} from {}...", model_id, model_dir_str);
        let t0 = Instant::now();

        let tokenizer = load_tokenizer(model_dir)?;

        let inner = match backend {
            LlmBackend::Qwen2 => {
                let model = qwen3_mlx::qwen2::load_qwen2_model(model_dir)
                    .map_err(|e| format!("Failed to load Qwen2 model: {:?}", e))?;
                LlmInner::Qwen2(model)
            }
            LlmBackend::Qwen3 => {
                let model = qwen3_mlx::load_model(model_dir)
                    .map_err(|e| format!("Failed to load Qwen3 model: {:?}", e))?;
                LlmInner::Qwen3(model)
            }
            LlmBackend::Mistral => {
                let model = mistral_mlx::load_model(model_dir)
                    .map_err(|e| format!("Failed to load Mistral model: {:?}", e))?;
                LlmInner::Mistral(model)
            }
            LlmBackend::Glm4 => {
                let model = glm4_mlx::load_model(model_dir)
                    .map_err(|e| format!("Failed to load GLM4 model: {:?}", e))?;
                LlmInner::Glm4(model)
            }
            LlmBackend::Mixtral => {
                let model = mixtral_mlx::load_model(model_dir)
                    .map_err(|e| format!("Failed to load Mixtral model: {:?}", e))?;
                LlmInner::Mixtral(model)
            }
            LlmBackend::MiniCpmSala => {
                let model = minicpm_sala_mlx::load_model(model_dir)
                    .map_err(|e| format!("Failed to load MiniCPM-SALA model: {:?}", e))?;
                LlmInner::MiniCpmSala(model)
            }
        };

        eprintln!("[llm] Model loaded in {:.1}s", t0.elapsed().as_secs_f64());

        Ok(LlmEngine {
            backend,
            model_id: model_id.to_string(),
            inner,
            tokenizer,
        })
    }

    /// Run chat completion inference.
    pub fn chat(
        &mut self,
        messages: &[(String, String)],
        max_tokens: usize,
        temperature: f32,
    ) -> Result<ChatResult, String> {
        match self.backend {
            LlmBackend::MiniCpmSala => self.chat_minicpm(messages, max_tokens, temperature),
            _ => self.chat_generic(messages, max_tokens, temperature),
        }
    }

    /// Generic chat for Qwen3/Mistral/GLM4/Mixtral using the Generate iterator.
    fn chat_generic(
        &mut self,
        messages: &[(String, String)],
        max_tokens: usize,
        temperature: f32,
    ) -> Result<ChatResult, String> {
        let prompt = format_chatml_prompt(messages);

        let encoding = self
            .tokenizer
            .encode(prompt.as_str(), false)
            .map_err(|e| format!("Tokenizer error: {}", e))?;
        let ids: Vec<u32> = encoding.get_ids().to_vec();
        let prompt_len = ids.len();

        let prompt_array = Array::from(&ids[..]).index(NewAxis);

        let t0 = Instant::now();
        let mut generated_ids: Vec<u32> = Vec::new();

        // Macro-like helper closure: run Generate iterator and collect results.
        // All backends share identical post-prefill logic.
        macro_rules! run_generate {
            ($generator:expr) => {{
                let prefill_done = Instant::now();
                for token_result in $generator.take(max_tokens) {
                    let token = token_result.map_err(|e| format!("Generate error: {:?}", e))?;
                    let token_id = token.item::<u32>();
                    if is_stop_token(token_id) {
                        break;
                    }
                    generated_ids.push(token_id);
                }
                let prefill_ms = (prefill_done - t0).as_secs_f64() * 1000.0;
                let decode_time = prefill_done.elapsed().as_secs_f64();
                let decode_tps = if generated_ids.is_empty() || decode_time == 0.0 {
                    0.0
                } else {
                    generated_ids.len() as f64 / decode_time
                };
                let text = self
                    .tokenizer
                    .decode(&generated_ids, true)
                    .map_err(|e| format!("Decode error: {}", e))?;
                Ok(ChatResult {
                    text,
                    prompt_tokens: prompt_len,
                    completion_tokens: generated_ids.len(),
                    prefill_ms,
                    decode_tps,
                })
            }};
        }

        match &mut self.inner {
            LlmInner::Qwen2(model) => {
                let mut cache = Vec::new();
                let generator = qwen3_mlx::qwen2::Generate::<qwen3_mlx::KVCache>::new(
                    model,
                    &mut cache,
                    temperature,
                    &prompt_array,
                );
                run_generate!(generator)
            }
            LlmInner::Qwen3(model) => {
                let mut cache = Vec::new();
                let generator = qwen3_mlx::Generate::<qwen3_mlx::KVCache>::new(
                    model,
                    &mut cache,
                    temperature,
                    &prompt_array,
                );
                run_generate!(generator)
            }
            LlmInner::Mistral(model) => {
                let mut cache = Vec::new();
                let generator = mistral_mlx::Generate::<mistral_mlx::KVCache>::new(
                    model,
                    &mut cache,
                    temperature,
                    &prompt_array,
                );
                run_generate!(generator)
            }
            LlmInner::Glm4(model) => {
                let mut cache = Vec::new();
                let generator = glm4_mlx::Generate::<glm4_mlx::KVCache>::new(
                    model,
                    &mut cache,
                    temperature,
                    &prompt_array,
                );
                run_generate!(generator)
            }
            LlmInner::Mixtral(model) => {
                let mut cache = Vec::new();
                let generator = mixtral_mlx::Generate::<mixtral_mlx::KVCache>::new(
                    model,
                    &mut cache,
                    temperature,
                    &prompt_array,
                );
                run_generate!(generator)
            }
            _ => unreachable!(),
        }
    }

    /// MiniCPM-SALA uses a different API with its own cache system.
    fn chat_minicpm(
        &mut self,
        messages: &[(String, String)],
        max_tokens: usize,
        temperature: f32,
    ) -> Result<ChatResult, String> {
        let model = match &mut self.inner {
            LlmInner::MiniCpmSala(m) => m,
            _ => unreachable!(),
        };

        let system_msg = messages
            .iter()
            .find(|(r, _)| r == "system")
            .map(|(_, c)| c.as_str())
            .unwrap_or("You are a helpful assistant.");

        let turns: Vec<(&str, &str)> = messages
            .iter()
            .filter(|(r, _)| r != "system")
            .map(|(r, c)| (r.as_str(), c.as_str()))
            .collect();

        let prompt = minicpm_sala_mlx::format_chat_prompt_multi(system_msg, &turns);

        let encoding = self
            .tokenizer
            .encode(prompt.as_str(), true)
            .map_err(|e| format!("Tokenizer error: {}", e))?;
        let prompt_tokens = encoding.get_ids();
        let prompt_len = prompt_tokens.len();

        let input = Array::from_slice(
            &prompt_tokens.iter().map(|&t| t as i32).collect::<Vec<_>>(),
            &[1, prompt_len as i32],
        );

        let mut caches = minicpm_sala_mlx::create_layer_caches(&model.args);

        // Prefill
        let t0 = Instant::now();
        let logits = model
            .forward(&input, &mut caches)
            .map_err(|e| format!("Forward error: {:?}", e))?;
        let last_logits = logits.index((.., -1, ..));
        let mut token = minicpm_sala_mlx::sample(&last_logits, temperature)
            .map_err(|e| format!("Sample error: {:?}", e))?;
        eval([&token]).map_err(|e| format!("Eval error: {:?}", e))?;
        let prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Decode
        let mut generated_ids: Vec<u32> = Vec::new();
        let decode_start = Instant::now();

        for _ in 0..max_tokens {
            let token_id = token.item::<u32>();
            if minicpm_sala_mlx::is_stop_token(token_id) {
                break;
            }
            generated_ids.push(token_id);

            let input = token
                .reshape(&[1, 1])
                .map_err(|e| format!("Reshape error: {:?}", e))?;
            let logits = model
                .forward(&input, &mut caches)
                .map_err(|e| format!("Forward error: {:?}", e))?;
            let last_logits = logits.index((.., -1, ..));
            token = minicpm_sala_mlx::sample(&last_logits, temperature)
                .map_err(|e| format!("Sample error: {:?}", e))?;
        }
        eval([&token]).map_err(|e| format!("Eval error: {:?}", e))?;

        let decode_time = decode_start.elapsed().as_secs_f64();
        let decode_tps = if generated_ids.is_empty() {
            0.0
        } else {
            generated_ids.len() as f64 / decode_time
        };

        let mut text = self
            .tokenizer
            .decode(&generated_ids, true)
            .map_err(|e| format!("Decode error: {}", e))?;

        // Strip thinking blocks
        text = minicpm_sala_mlx::strip_thinking(&text).to_string();

        Ok(ChatResult {
            text,
            prompt_tokens: prompt_len,
            completion_tokens: generated_ids.len(),
            prefill_ms,
            decode_tps,
        })
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn backend(&self) -> LlmBackend {
        self.backend
    }
}

fn load_tokenizer(model_dir: &Path) -> Result<tokenizers::Tokenizer, String> {
    let path = model_dir.join("tokenizer.json");
    tokenizers::Tokenizer::from_file(&path)
        .map_err(|e| format!("Failed to load tokenizer: {}", e))
}

fn format_chatml_prompt(messages: &[(String, String)]) -> String {
    let mut prompt = String::new();
    for (role, content) in messages {
        prompt.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", role, content));
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

fn is_stop_token(token_id: u32) -> bool {
    // Common stop tokens across models
    token_id == 2           // EOS
        || token_id == 151643   // Qwen <|endoftext|>
        || token_id == 151645   // Qwen <|im_end|>
        || token_id == 73440    // MiniCPM <|im_end|>
}
