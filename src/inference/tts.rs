use std::path::Path;
use std::time::Instant;

/// TTS engine using GPT-SoVITS voice cloning.
pub struct TtsEngine {
    cloner: gpt_sovits_mlx::VoiceCloner,
    model_id: String,
}

impl TtsEngine {
    /// Load GPT-SoVITS model.
    pub fn load(_model_dir: &Path, model_id: &str) -> Result<Self, String> {
        eprintln!("[tts] Loading GPT-SoVITS...");
        let t0 = Instant::now();

        let cloner = gpt_sovits_mlx::VoiceCloner::with_defaults()
            .map_err(|e| format!("Failed to load TTS model: {:?}", e))?;

        eprintln!("[tts] Model loaded in {:.1}s", t0.elapsed().as_secs_f64());

        Ok(TtsEngine {
            cloner,
            model_id: model_id.to_string(),
        })
    }

    /// Set reference audio for voice cloning.
    pub fn set_reference(&mut self, audio_path: &str) -> Result<(), String> {
        self.cloner
            .set_reference_audio(audio_path)
            .map_err(|e| format!("Reference audio error: {:?}", e))
    }

    /// Synthesize speech from text.
    pub fn synthesize(&mut self, text: &str) -> Result<Vec<f32>, String> {
        let t0 = Instant::now();

        let audio = self
            .cloner
            .synthesize(text)
            .map_err(|e| format!("Synthesis error: {:?}", e))?;

        let samples: Vec<f32> = audio.samples.clone();

        let duration_s = samples.len() as f64 / audio.sample_rate as f64;
        let elapsed = t0.elapsed().as_secs_f64();
        let rtf = if elapsed > 0.0 { duration_s / elapsed } else { 0.0 };

        eprintln!(
            "[tts] Synthesized {:.1}s audio in {:.2}s ({:.1}x real-time)",
            duration_s, elapsed, rtf
        );

        Ok(samples)
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }
}
