use anyhow::Result;
use chess_ai_rust::engine::ExplainableEngine;
use chess_ai_rust::features::extract_features;
use chess_ai_rust::ml::{train_surrogate_model, PhaseEnsemble, SurrogateExplainer};
use clap::{Parser, Subcommand};
use shakmaty::{Chess, Position};
use std::io::{self, Write};
use std::path::Path;

#[derive(Parser)]
#[command(name = "chess-ai")]
#[command(about = "Explainable Chess Engine in Rust", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Play an interactive chess game
    Play {
        #[arg(short, long, default_value = "stockfish")]
        stockfish_path: String,
        #[arg(short, long, default_value_t = 12)]
        depth: u32,
    },
    /// Run a feature explainability audit
    Audit {
        #[arg(short, long, default_value = "stockfish")]
        stockfish_path: String,
        #[arg(short, long)]
        fen: Option<String>,
        #[arg(short, long, default_value = "model.json")]
        model_path: String,
    },
    /// Train the surrogate model
    Train {
        #[arg(short, long, default_value = "stockfish")]
        stockfish_path: String,
        #[arg(short, long, default_value = "model.json")]
        output_path: String,
        #[arg(short, long, default_value_t = 100)]
        n_positions: usize,
    },
    /// Start the explainable chess web server
    Server {
        #[arg(short, long, default_value = "stockfish")]
        stockfish_path: String,
        #[arg(short = 'H', long, default_value = "127.0.0.1")]
        host: String,
        #[arg(short, long, default_value_t = 5000)]
        port: u16,
    },
    /// Syzygy tablebase utilities
    Syzygy {
        #[command(subcommand)]
        action: SyzygyAction,
    },
    /// Check environment for dependencies
    Doctor,
}

#[derive(Subcommand)]
enum SyzygyAction {
    /// Download 3-5 piece tablebases
    Download {
        #[arg(short, long, default_value = "~/syzygy")]
        dest: String,
    },
    /// Verify tablebase integration
    Verify {
        #[arg(short, long, default_value = "stockfish")]
        stockfish_path: String,
        #[arg(long, default_value = "~/syzygy")]
        syzygy_path: String,
        #[arg(short, long, default_value = "model.json")]
        model_path: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Play {
            stockfish_path,
            depth,
        } => {
            let mut engine = ExplainableEngine::new(&stockfish_path)?;
            println!("Welcome to the Explainable Chess Engine (Rust Edition)!");

            loop {
                print!("Your move (UCI): ");
                io::stdout().flush()?;

                let mut input = String::new();
                io::stdin().read_line(&mut input)?;
                let input = input.trim();

                if input == "quit" || input == "exit" {
                    break;
                }

                if let Err(e) = engine.make_move(input) {
                    println!("Error: {}", e);
                    continue;
                }

                println!("Stockfish is thinking...");
                let best_move = engine.get_best_move(depth)?;
                println!("Stockfish plays: {}", best_move);
                engine.make_move(&best_move)?;
            }
        }
        Commands::Audit {
            stockfish_path,
            fen,
            model_path,
        } => {
            let fen_str = fen.unwrap_or_else(|| {
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".to_string()
            });
            let mut engine = ExplainableEngine::new(&stockfish_path)?;

            let pos: Chess = fen_str
                .parse::<shakmaty::fen::Fen>()?
                .into_position(shakmaty::CastlingMode::Standard)?;

            println!("Auditing FEN: {}", fen_str);
            let feats = extract_features(&pos);

            println!("\nExtracted Features:");
            for (name, val) in feats {
                println!("  {:30}: {:>8.3}", name, val);
            }

            println!("Stockfish is thinking...");
            let best_move = engine.get_best_move(12)?;
            println!("\nEngine Recommendation: {}", best_move);

            if Path::new(&model_path).exists() {
                println!("\nLoading model from {}...", model_path);
                let model_str = std::fs::read_to_string(&model_path)?;
                let model: PhaseEnsemble = serde_json::from_str(&model_str)?;
                let explainer = SurrogateExplainer::new(model);

                // For audit, we simulate a move to see explanations.
                // Let's show explanations for the recommended best move.
                let mut pos_after = pos.clone();
                let uci_move: shakmaty::uci::UciMove = best_move.parse()?;
                if let Ok(m) = uci_move.to_move(&pos) {
                    pos_after.play_unchecked(m);
                    let feats_after = extract_features(&pos_after); // This is just absolute, usually we'd want delta
                                                                    // However, our explainer takes 'features_after' (which are usually already deltas in the Python code)
                                                                    // Let's match the Python logic: explainer calculates delta if needed.
                                                                    // Actually, our Rust explainer takes 'features_after' and calculates delta from 'model.feature_names'.

                    let reasons = explainer.explain_move(&feats_after, 5, 0.05);
                    println!("\nMove Explanations (for {}):", best_move);
                    for (_, cp, text) in reasons {
                        println!("  - {} ({:+.1} cp)", text, cp);
                    }
                }
            } else {
                println!(
                    "\n[Note] Model file not found at {}. Skipping ML explanations.",
                    model_path
                );
            }
        }
        Commands::Train {
            stockfish_path,
            output_path,
            n_positions,
        } => {
            println!(
                "Starting surrogate model training ({} positions)...",
                n_positions
            );
            let ensemble = train_surrogate_model(&stockfish_path, n_positions)?;
            let json = serde_json::to_string_pretty(&ensemble)?;
            std::fs::write(&output_path, json)?;
            println!("✅ Model saved to {}", output_path);
        }
        Commands::Server {
            stockfish_path,
            host,
            port,
        } => {
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(async {
                chess_ai_rust::web_server::start_server(stockfish_path, host, port).await
            })?;
        }
        Commands::Syzygy { action } => match action {
            SyzygyAction::Download { dest } => {
                let expanded_dest = if dest.starts_with("~/") {
                    dest.replace("~", &std::env::var("HOME")?)
                } else {
                    dest
                };
                chess_ai_rust::syzygy_utils::download_syzygy(&expanded_dest)?;
            }
            SyzygyAction::Verify {
                stockfish_path,
                syzygy_path,
                model_path,
            } => {
                let expanded_path = if syzygy_path.starts_with("~/") {
                    syzygy_path.replace("~", &std::env::var("HOME")?)
                } else {
                    syzygy_path
                };
                chess_ai_rust::syzygy_utils::verify_syzygy(
                    &stockfish_path,
                    &expanded_path,
                    Some(&model_path),
                )?;
            }
        },
        Commands::Doctor => {
            println!("Checking environment dependencies...");
            // We use a default path or the environment variable
            let stockfish_path = std::env::var("STOCKFISH_PATH").unwrap_or_else(|_| "stockfish".to_string());
            match ExplainableEngine::new(&stockfish_path) {
                Ok(_) => println!("✅ Stockfish found and responding at '{}'.", stockfish_path),
                Err(e) => {
                    println!("❌ Stockfish error: {}", e);
                    println!("   Please ensure Stockfish is in your PATH or set the 'STOCKFISH_PATH' environment variable.");
                }
            }
        }
    }

    Ok(())
}
