use std::path::Path;
use std::time::Instant;

use mlx_rs::ops::indexing::{IndexOp, NewAxis};
use mlx_rs::transforms::eval;
use mlx_rs::Array;

use super::ChatResult;

/// Vision-Language Model engine (Moxin-7B VLM).
pub struct VlmEngine {
    model: moxin_vlm_mlx::MoxinVLM,
    tokenizer: tokenizers::Tokenizer,
    model_id: String,
}

impl VlmEngine {
    /// Load a VLM model from a directory.
    pub fn load(model_dir: &Path, model_id: &str) -> Result<Self, String> {
        eprintln!("[vlm] Loading {} from {}...", model_id, model_dir.display());
        let t0 = Instant::now();

        let mut model = moxin_vlm_mlx::load_model(model_dir)
            .map_err(|e| format!("Failed to load VLM: {:?}", e))?;

        // Apply 8-bit quantization
        let model = model
            .quantize(64, 8)
            .map_err(|e| format!("Quantization failed: {:?}", e))?;

        let tokenizer = moxin_vlm_mlx::load_tokenizer(model_dir)
            .map_err(|e| format!("Tokenizer error: {:?}", e))?;

        eprintln!("[vlm] Model loaded in {:.1}s", t0.elapsed().as_secs_f64());

        Ok(VlmEngine {
            model,
            tokenizer,
            model_id: model_id.to_string(),
        })
    }

    /// Run VLM inference with image and text prompt.
    pub fn chat_with_image(
        &mut self,
        image_data: &[u8],
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> Result<ChatResult, String> {
        let t0 = Instant::now();

        // Decode image to 224x224x3 float tensor
        let img = image::load_from_memory(image_data)
            .map_err(|e| format!("Image decode error: {}", e))?
            .resize_exact(224, 224, image::imageops::FilterType::Lanczos3)
            .to_rgb8();

        let pixels: Vec<f32> = img.pixels().flat_map(|p| {
            p.0.iter().map(|&v| v as f32 / 255.0)
        }).collect();

        let img_tensor = Array::from_slice(&pixels, &[1, 224, 224, 3]);

        let dino_img = moxin_vlm_mlx::normalize_dino(&img_tensor)
            .map_err(|e| format!("DINOv2 normalize error: {:?}", e))?;
        let siglip_img = moxin_vlm_mlx::normalize_siglip(&img_tensor)
            .map_err(|e| format!("SigLIP normalize error: {:?}", e))?;

        // Format prompt with image token
        let full_prompt = format!("<image>\n{}", prompt);
        let encoding = self
            .tokenizer
            .encode(full_prompt.as_str(), true)
            .map_err(|e| format!("Tokenizer error: {}", e))?;
        let ids = encoding.get_ids();
        let prompt_len = ids.len();

        let input_ids = Array::from_slice(
            &ids.iter().map(|&t| t as i32).collect::<Vec<_>>(),
            &[1, prompt_len as i32],
        );

        let mut cache = Vec::new();
        let mut generated_ids: Vec<u32> = Vec::new();

        let generator = moxin_vlm_mlx::Generate::<moxin_vlm_mlx::KVCache>::new(
            &mut self.model,
            &mut cache,
            temperature,
            dino_img,
            siglip_img,
            input_ids,
        );

        let prefill_done = Instant::now();
        for token_result in generator.take(max_tokens) {
            let token = token_result.map_err(|e| format!("Generate error: {:?}", e))?;
            let token_id = token.item::<u32>();
            if token_id == 2 || token_id == 32000 {
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
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }
}
