//! Subtitle text set with cosmic-text, which shapes every script (Arabic
//! letters join, Arabic and Hebrew run right to left, Indic and Thai
//! syllables cluster) and falls back to any installed font that has the
//! characters. egui's own text does neither.

use cosmic_text::{Align, Attrs, Buffer, Color, Family, FontSystem, Metrics, Shaping, SwashCache};
use std::collections::HashSet;

pub struct SubtitleText {
    fonts: FontSystem,
    glyphs: SwashCache,
    /// Characters already reported as missing from every font.
    missing: HashSet<char>,
}

impl SubtitleText {
    /// Inter and the system's fonts, `~/.local/share/fonts` included. Reading
    /// them takes a moment, so this runs off the frame loop.
    pub fn load() -> SubtitleText {
        let mut fonts = FontSystem::new();
        fonts
            .db_mut()
            .load_font_data(super::theme::INTER_REGULAR.to_vec());
        log::info!("subtitle fonts: {} faces", fonts.db().len());
        SubtitleText {
            fonts,
            glyphs: SwashCache::new(),
            missing: HashSet::new(),
        }
    }

    /// White text in centred lines at most `width` pixels wide, `size`
    /// pixels high: as wide as its ink, as tall as its lines (and any mark
    /// reaching beyond them).
    pub fn render(&mut self, text: &str, size: f32, width: f32) -> egui::ColorImage {
        let mut buffer = Buffer::new(&mut self.fonts, Metrics::new(size, size * 1.25));
        buffer.set_size(Some(width), None);
        buffer.set_text(
            text,
            &Attrs::new().family(Family::Name("Inter")),
            Shaping::Advanced,
            Some(Align::Center),
        );
        buffer.shape_until_scroll(&mut self.fonts, false);
        self.report_missing(&buffer);

        let lines = buffer
            .layout_runs()
            .map(|r| r.line_top + r.line_height)
            .fold(0.0, f32::max)
            .ceil() as usize;
        // Room for marks above and below the lines (Arabic, Vietnamese).
        let margin = (size * 0.5).ceil() as usize;
        let (w, h) = (width.ceil() as usize, lines + 2 * margin);
        let mut alpha = vec![0u8; w * h];
        buffer.draw(
            &mut self.fonts,
            &mut self.glyphs,
            Color::rgb(255, 255, 255),
            |x, y, gw, gh, color| {
                let a = color.a() as u32;
                let y = y + margin as i32;
                for py in y.max(0)..(y + gh as i32).min(h as i32) {
                    for px in x.max(0)..(x + gw as i32).min(w as i32) {
                        let d = &mut alpha[py as usize * w + px as usize];
                        // Joined letters overlap: lay coverage over coverage.
                        *d = (a + *d as u32 * (255 - a) / 255) as u8;
                    }
                }
            },
        );
        crop(&alpha, w, margin..margin + lines)
    }

    /// Logs characters no installed font has, each once: they show as
    /// boxes until a font covering them is added.
    fn report_missing(&mut self, buffer: &Buffer) {
        let mut new = String::new();
        for run in buffer.layout_runs() {
            for g in run.glyphs.iter().filter(|g| g.glyph_id == 0) {
                new.extend(
                    run.text[g.start..g.end]
                        .chars()
                        .filter(|c| !c.is_whitespace() && self.missing.insert(*c)),
                );
            }
        }
        // A film in a script without a font has thousands of them.
        if !new.is_empty() && self.missing.len() <= 200 {
            log::warn!(
                "subtitles: no installed font has {new:?}; add one that does to \
                 ~/.local/share/fonts"
            );
        }
    }
}

/// White pixels of the given coverage, cut to the covered columns and to
/// `rows` plus any covered rows beyond them.
fn crop(alpha: &[u8], w: usize, rows: std::ops::Range<usize>) -> egui::ColorImage {
    let h = alpha.len() / w.max(1);
    let covered = |x: usize, y: usize| alpha[y * w + x] > 0;
    let inked: Vec<usize> = (0..h).filter(|&y| (0..w).any(|x| covered(x, y))).collect();
    let cols: Vec<usize> = (0..w).filter(|&x| (0..h).any(|y| covered(x, y))).collect();
    let (Some(&first), Some(&last), Some(&left), Some(&right)) =
        (inked.first(), inked.last(), cols.first(), cols.last())
    else {
        return egui::ColorImage::new([0, 0], Vec::new());
    };
    let (top, bottom) = (first.min(rows.start), last.max(rows.end - 1));
    let (cw, ch) = (right - left + 1, bottom - top + 1);
    let mut rgba = Vec::with_capacity(cw * ch * 4);
    for y in top..=bottom {
        for &a in &alpha[y * w + left..=y * w + right] {
            rgba.extend_from_slice(&[255, 255, 255, a]);
        }
    }
    egui::ColorImage::from_rgba_unmultiplied([cw, ch], &rgba)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Only the bundled Inter: the system's fonts differ between machines.
    fn inter_only() -> SubtitleText {
        let mut db = cosmic_text::fontdb::Database::new();
        db.load_font_data(super::super::theme::INTER_REGULAR.to_vec());
        SubtitleText {
            fonts: FontSystem::new_with_locale_and_db("en-US".into(), db),
            glyphs: SwashCache::new(),
            missing: HashSet::new(),
        }
    }

    #[test]
    fn renders_cropped_and_wrapped() {
        let mut t = inter_only();
        let one = t.render("Hello world", 40.0, 1000.0);
        assert!(one.size[0] > 100 && one.size[0] < 400, "{:?}", one.size);
        // One line box, whatever the ink.
        assert_eq!(one.size[1], 50);
        assert_eq!(t.render("ace", 40.0, 1000.0).size[1], 50);
        let two = t.render("Hello world, a line too long to fit", 40.0, 300.0);
        assert!(two.size[0] <= 300, "{:?}", two.size);
        assert!(
            two.size[1] >= 100 && two.size[1].is_multiple_of(50),
            "{:?}",
            two.size
        );
        assert_eq!(t.render(" ", 40.0, 300.0).size, [0, 0]);
    }

    #[test]
    fn missing_characters_are_noted() {
        let mut t = inter_only();
        t.render("مرحبا", 40.0, 600.0);
        assert!(t.missing.contains(&'م'), "{:?}", t.missing);
        assert!(!t.missing.contains(&'a'));
    }
}
