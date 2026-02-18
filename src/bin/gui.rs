use mofa_local_llm::gui::app::MofaApp;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1024.0, 768.0])
            .with_title("MOFA Local LLM"),
        ..Default::default()
    };

    eframe::run_native(
        "MOFA Local LLM",
        options,
        Box::new(|cc| {
            // Load Chinese font
            let mut fonts = egui::FontDefinitions::default();
            for font_path in &[
                "/System/Library/Fonts/Hiragino Sans GB.ttc",
                "/System/Library/Fonts/STHeiti Light.ttc",
                "/System/Library/Fonts/PingFang.ttc",
            ] {
                if let Ok(font_data) = std::fs::read(font_path) {
                    fonts.font_data.insert(
                        "chinese".to_string(),
                        egui::FontData::from_owned(font_data),
                    );
                    fonts
                        .families
                        .entry(egui::FontFamily::Proportional)
                        .or_default()
                        .push("chinese".to_string());
                    break;
                }
            }
            cc.egui_ctx.set_fonts(fonts);

            Ok(Box::new(MofaApp::new(cc)))
        }),
    )
}
