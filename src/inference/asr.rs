use std::path::Path;
use std::time::Instant;

/// ASR engine using FunASR Paraformer.
pub struct AsrEngine {
    model: funasr_mlx::Paraformer,
    vocab: funasr_mlx::Vocabulary,
    model_id: String,
}

impl AsrEngine {
    /// Load FunASR Paraformer model.
    pub fn load(model_dir: &Path, model_id: &str) -> Result<Self, String> {
        eprintln!("[asr] Loading {} from {}...", model_id, model_dir.display());
        let t0 = Instant::now();

        let weights_path = model_dir.join("paraformer.safetensors");
        let cmvn_path = model_dir.join("am.mvn");
        let vocab_path = if model_dir.join("tokens.txt").exists() {
            model_dir.join("tokens.txt")
        } else {
            model_dir.join("vocab.txt")
        };

        let mut model = funasr_mlx::load_model(&weights_path)
            .map_err(|e| format!("Failed to load ASR model: {:?}", e))?;

        if cmvn_path.exists() {
            let (addshift, rescale) = funasr_mlx::parse_cmvn_file(&cmvn_path)
                .map_err(|e| format!("CMVN parse error: {:?}", e))?;
            model.set_cmvn(addshift, rescale);
        }

        let vocab = funasr_mlx::Vocabulary::load(&vocab_path)
            .map_err(|e| format!("Vocabulary load error: {:?}", e))?;

        eprintln!(
            "[asr] Model loaded in {:.1}s (vocab size: {})",
            t0.elapsed().as_secs_f64(),
            vocab.len()
        );

        Ok(AsrEngine {
            model,
            vocab,
            model_id: model_id.to_string(),
        })
    }

    /// Transcribe audio samples (16kHz, mono, f32 [-1, 1]).
    pub fn transcribe(&mut self, samples: &[f32]) -> Result<String, String> {
        let t0 = Instant::now();

        let text = funasr_mlx::transcribe(&mut self.model, samples, &self.vocab)
            .map_err(|e| format!("Transcription error: {:?}", e))?;

        let duration_s = samples.len() as f64 / 16000.0;
        let elapsed = t0.elapsed().as_secs_f64();
        let rtf = if elapsed > 0.0 { duration_s / elapsed } else { 0.0 };

        eprintln!(
            "[asr] Transcribed {:.1}s audio in {:.2}s ({:.1}x real-time)",
            duration_s, elapsed, rtf
        );

        Ok(text)
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }
}

/// Resample audio to 16kHz mono.
pub fn resample_to_16khz(samples: &[f32], from_rate: u32) -> Vec<f32> {
    if from_rate == 16000 {
        return samples.to_vec();
    }

    let ratio = 16000.0 / from_rate as f64;
    let new_len = (samples.len() as f64 * ratio) as usize;
    let mut resampled = Vec::with_capacity(new_len);

    for i in 0..new_len {
        let src_idx = i as f64 / ratio;
        let idx0 = src_idx.floor() as usize;
        let idx1 = (idx0 + 1).min(samples.len() - 1);
        let frac = src_idx - idx0 as f64;
        let val = samples[idx0] as f64 * (1.0 - frac) + samples[idx1] as f64 * frac;
        resampled.push(val as f32);
    }

    resampled
}
