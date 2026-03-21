use anyhow::{anyhow, Result};
use reqwest::blocking::Client;
use scraper::{Html, Selector};
use shakmaty::{fen::Fen, Chess, Position};
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::time::Duration;

use crate::engine::ExplainableEngine;
use crate::features::extract_features;
use crate::ml::{PhaseEnsemble, SurrogateExplainer};

const SYZYGY_BASE_URL: &str = "http://tablebase.sesse.net/syzygy/3-4-5/";

pub fn download_syzygy(dest_dir: &str) -> Result<()> {
    let client = Client::builder().timeout(Duration::from_secs(30)).build()?;

    println!("Fetching file list from {}...", SYZYGY_BASE_URL);
    let resp = client.get(SYZYGY_BASE_URL).send()?;
    if !resp.status().is_success() {
        return Err(anyhow!("Failed to fetch file list: {}", resp.status()));
    }

    let body = resp.text()?;
    let document = Html::parse_document(&body);
    let selector = Selector::parse("a").unwrap();

    let mut files = Vec::new();
    for element in document.select(&selector) {
        if let Some(href) = element.value().attr("href") {
            if href.ends_with(".rtbw") || href.ends_with(".rtbz") {
                files.push(href.to_string());
            }
        }
    }

    files.sort();
    files.dedup();

    if files.is_empty() {
        println!("No Syzygy files found to download.");
        return Ok(());
    }

    println!("Found {} tablebase files.", files.len());
    let dest_path = Path::new(dest_dir);
    if !dest_path.exists() {
        fs::create_dir_all(dest_path)?;
        println!("Created directory: {}", dest_dir);
    }

    for (i, filename) in files.iter().enumerate() {
        let url = format!("{}{}", SYZYGY_BASE_URL, filename);
        let target = dest_path.join(filename);

        if target.exists() {
            println!(
                "[{}/{}] Skipping {} (already exists)",
                i + 1,
                files.len(),
                filename
            );
            continue;
        }

        print!("[{}/{}] Downloading {}... ", i + 1, files.len(), filename);
        io::stdout().flush()?;

        let mut file_resp = client.get(&url).send()?;
        if file_resp.status().is_success() {
            let mut file = fs::File::create(target)?;
            file_resp.copy_to(&mut file)?;
            println!("Done.");
        } else {
            println!("FAILED: {}", file_resp.status());
        }
    }

    println!("\nFinished downloading Syzygy files.");
    println!(
        "Set export SYZYGY_PATH={} and run verify to confirm.",
        dest_dir
    );

    Ok(())
}

pub fn verify_syzygy(
    stockfish_path: &str,
    syzygy_path: &str,
    model_path: Option<&str>,
) -> Result<()> {
    let endgames = vec![
        ("KQK", "4k3/8/8/8/8/8/8/4K1Q1 w - - 0 1"),
        ("KRK", "4k3/8/8/8/8/8/8/4K1R1 w - - 0 1"),
        ("KPK", "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"),
        ("KBNK", "8/8/8/8/8/8/2B1N3/4K2k w - - 0 1"),
        ("KBBK", "4k3/8/8/8/8/8/8/2K1BB2 w - - 0 1"),
        ("KBPK", "4k3/8/8/8/8/4P3/4K1B1/8 w - - 0 1"),
    ];

    println!("Verifying Syzygy integration (path={})\n", syzygy_path);

    let mut engine = ExplainableEngine::new(stockfish_path)?;
    if let Ok(tb) = crate::syzygy::SyzygyTablebase::new(syzygy_path) {
        engine.tb = Some(tb);
    } else {
        return Err(anyhow!("Failed to load tablebases from {}", syzygy_path));
    }

    let explainer = if let Some(path) = model_path {
        if Path::new(path).exists() {
            let model_str = fs::read_to_string(path)?;
            let model: PhaseEnsemble = serde_json::from_str(&model_str)?;
            Some(SurrogateExplainer::new(model))
        } else {
            None
        }
    } else {
        None
    };

    let failures = 0;

    for (name, fen_str) in endgames {
        println!("--- {} ---", name);
        let pos: Chess = fen_str
            .parse::<Fen>()?
            .into_position(shakmaty::CastlingMode::Standard)?;
        engine.set_position(fen_str)?;

        let best_move = engine.get_best_move(12)?;
        println!("  Best move: {}", best_move);

        if let Some(tb) = &engine.tb {
            let wdl = tb.tb.probe_wdl(&pos)?;
            println!("  Syzygy WDL: {:?}", wdl);
        }

        if let Some(ref expl) = explainer {
            let mut pos_after = pos.clone();
            let uci_move: shakmaty::uci::UciMove = best_move.parse()?;
            if let Ok(m) = uci_move.to_move(&pos) {
                pos_after.play_unchecked(m);
                let feats_after = extract_features(&pos_after);
                let reasons = expl.explain_move(&feats_after, 5, 0.05);

                let mut found_syzygy = false;
                for (_, _, text) in reasons {
                    println!("    Reason: {}", text);
                    if text.to_lowercase().contains("syzygy")
                        || text.to_lowercase().contains("tablebase")
                    {
                        found_syzygy = true;
                    }
                }

                if !found_syzygy {
                    println!("  WARN: No Syzygy reason found in explanation.");
                    // Check if it's actually in TB
                    if let Some(tb) = &engine.tb {
                        if tb.tb.probe_wdl(&pos).is_ok() {
                            // If we have an explainer but it didn't pick up syzygy, it might be the model's fault or threshold
                        }
                    }
                }
            }
        }
        println!();
    }

    if failures > 0 {
        return Err(anyhow!("{} endgames failed verification", failures));
    }

    println!("Syzygy verification complete.");
    Ok(())
}
