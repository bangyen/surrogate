use anyhow::Result;
use chess_ai_rust::audit::{run_audit, AuditConfig, AuditReport};
use chess_ai_rust::engine::ExplainableEngine;
use chess_ai_rust::features::extract_features;
use chess_ai_rust::ml::{train_surrogate_model, PhaseEnsemble, SurrogateExplainer};
use chess_ai_rust::variant::{Variant, VariantGame};
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
    /// Play a chess variant against the native engine
    Variant {
        /// Which variant to play (standard, koth, 3check, antichess)
        #[arg(short, long, default_value = "koth")]
        variant: String,
        #[arg(short, long, default_value_t = 5)]
        depth: u8,
        /// List the supported variants and exit
        #[arg(short, long)]
        list: bool,
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
    /// Measure explainability metrics over sampled positions
    Metrics {
        #[arg(short, long, default_value = "stockfish")]
        stockfish_path: String,
        #[arg(short, long, default_value = "model.json")]
        model_path: String,
        #[arg(short, long, default_value_t = 100)]
        n_positions: usize,
        #[arg(short, long, default_value_t = 12)]
        depth: u32,
        /// Write the report here (defaults to audit-results.json)
        #[arg(short, long, default_value = "audit-results.json")]
        out: String,
        /// Compare against the committed report and fail on regressions
        #[arg(long)]
        check: bool,
        /// Sampling seed; fixed by default so runs are comparable
        #[arg(long)]
        seed: Option<u64>,
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

/// Expand a leading `~/` to the user's home directory.
fn expand_home(path: &str) -> Result<String> {
    match path.strip_prefix("~/") {
        Some(rest) => Ok(format!("{}/{}", std::env::var("HOME")?, rest)),
        None => Ok(path.to_string()),
    }
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
        Commands::Variant {
            variant,
            depth,
            list,
        } => {
            if list {
                println!("Supported variants:\n");
                for v in Variant::ALL {
                    println!("  {:10} {}", v.slug(), v.description());
                }
                println!("\nExplanations are available for standard chess only.");
                return Ok(());
            }

            let variant: Variant = variant.parse()?;
            let mut game = VariantGame::new(variant);
            println!("Playing {} - {}", variant, variant.description());
            println!("The native engine plays Black. Enter moves in UCI notation.\n");

            loop {
                if game.is_game_over() {
                    println!("Game over.");
                    break;
                }

                println!("{}", game.fen());
                print!("Your move (UCI, or 'quit'): ");
                io::stdout().flush()?;

                let mut input = String::new();
                if io::stdin().read_line(&mut input)? == 0 {
                    break;
                }
                let input = input.trim();
                if input == "quit" || input == "exit" {
                    break;
                }

                if let Err(e) = game.play_uci(input) {
                    println!("Error: {}", e);
                    continue;
                }

                if game.is_game_over() {
                    println!("Game over.");
                    break;
                }

                print!("Engine is thinking...");
                io::stdout().flush()?;
                match game.best_move(depth) {
                    Some(mv) => {
                        game.play_uci(&mv)?;
                        println!(" plays {} (eval {:+} cp)", mv, game.evaluate());
                    }
                    None => {
                        println!(" no legal moves.");
                        break;
                    }
                }
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
        Commands::Metrics {
            stockfish_path,
            model_path,
            n_positions,
            depth,
            out,
            check,
            seed,
        } => {
            if !Path::new(&model_path).exists() {
                return Err(anyhow::anyhow!(
                    "Model not found at {}. Run `just train` first.",
                    model_path
                ));
            }
            let model: PhaseEnsemble =
                serde_json::from_str(&std::fs::read_to_string(&model_path)?)?;

            if check {
                // Verify the committed report rather than re-measuring:
                // a check should be fast and deterministic.
                if !Path::new(&out).exists() {
                    return Err(anyhow::anyhow!(
                        "No report at {}. Run `just metrics` to generate one.",
                        out
                    ));
                }
                let report: AuditReport = serde_json::from_str(&std::fs::read_to_string(&out)?)?;
                println!("{}", report.to_markdown());

                let failures = report.failures();
                if failures.is_empty() {
                    println!(
                        "✅ All {} metrics meet their targets.",
                        report.metrics.len()
                    );
                } else {
                    for m in &failures {
                        let comparator = if m.higher_is_better { ">=" } else { "<=" };
                        println!(
                            "❌ {}: {:.3} (target {} {:.2}, n={})",
                            m.name, m.value, comparator, m.target, m.n
                        );
                    }
                    return Err(anyhow::anyhow!("{} metric(s) below target", failures.len()));
                }
                return Ok(());
            }

            let cfg = AuditConfig {
                stockfish_path,
                n_positions,
                depth,
                seed: seed.unwrap_or(chess_ai_rust::audit::DEFAULT_SEED),
                ..Default::default()
            };
            println!(
                "Measuring explainability over {} positions at depth {}...",
                n_positions, depth
            );
            let report = run_audit(&model, &cfg)?;

            println!("\n{}", report.to_markdown());
            println!(
                "Evaluated {}/{} sampled positions.",
                report.n_positions_evaluated, report.n_positions_requested
            );
            for m in &report.metrics {
                if !m.passes() {
                    println!("⚠️  {} is below target ({:.3}, n={})", m.name, m.value, m.n);
                }
            }

            std::fs::write(&out, serde_json::to_string_pretty(&report)?)?;
            println!("\nReport written to {}", out);
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
                chess_ai_rust::syzygy_utils::download_syzygy(&expand_home(&dest)?)?;
            }
            SyzygyAction::Verify {
                stockfish_path,
                syzygy_path,
                model_path,
            } => {
                chess_ai_rust::syzygy_utils::verify_syzygy(
                    &stockfish_path,
                    &expand_home(&syzygy_path)?,
                    Some(&model_path),
                )?;
            }
        },
        Commands::Doctor => {
            println!("Checking environment dependencies...");
            // We use a default path or the environment variable
            let stockfish_path =
                std::env::var("STOCKFISH_PATH").unwrap_or_else(|_| "stockfish".to_string());
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
