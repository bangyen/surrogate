pub mod search;
pub mod eval;
pub mod see;
pub mod zobrist;

use anyhow::{anyhow, Result};
use shakmaty::{fen::Fen, Chess, Move, Position};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::time::Duration;

pub struct UciEngine {
    process: Child,
}

impl UciEngine {
    pub fn new(path: &str) -> Result<Self> {
        let process = Command::new(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()?;

        let mut engine = UciEngine { process };
        engine.send_command("uci")?;
        engine.wait_for_line("uciok", Duration::from_secs(5))?;
        Ok(engine)
    }

    pub fn send_command(&mut self, cmd: &str) -> Result<()> {
        let stdin = self
            .process
            .stdin
            .as_mut()
            .ok_or_else(|| anyhow!("Failed to open stdin"))?;
        writeln!(stdin, "{}", cmd)?;
        stdin.flush()?;
        Ok(())
    }

    pub fn is_ready(&mut self) -> Result<()> {
        self.send_command("isready")?;
        self.wait_for_line("readyok", Duration::from_secs(5))?;
        Ok(())
    }

    pub fn wait_for_line(&mut self, expected: &str, _timeout: Duration) -> Result<String> {
        let stdout = self
            .process
            .stdout
            .as_mut()
            .ok_or_else(|| anyhow!("Failed to open stdout"))?;
        let mut reader = BufReader::new(stdout);

        loop {
            let mut line = String::new();
            if reader.read_line(&mut line)? == 0 {
                return Err(anyhow!("EOF waiting for {}", expected));
            }
            if line.contains(expected) {
                return Ok(line);
            }
        }
    }

    pub fn get_best_move(&mut self, fen: &str, depth: u32) -> Result<String> {
        self.is_ready()?;
        self.send_command(&format!("position fen {}", fen))?;
        self.send_command(&format!("go depth {}", depth))?;
        let line = self.wait_for_line("bestmove", Duration::from_secs(30))?;

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 2 && parts[0] == "bestmove" {
            Ok(parts[1].to_string())
        } else {
            Err(anyhow!("Unexpected response from engine: {}", line))
        }
    }

    pub fn get_evaluation(&mut self, fen: &str, depth: u32) -> Result<i32> {
        self.is_ready()?;
        self.send_command(&format!("position fen {}", fen))?;
        self.send_command(&format!("go depth {}", depth))?;

        let mut last_score = 0;
        let stdout = self.process.stdout.as_mut().unwrap();
        let mut reader = BufReader::new(stdout);

        loop {
            let mut line = String::new();
            if reader.read_line(&mut line)? == 0 {
                break;
            }

            if line.contains("score cp") || line.contains("score mate") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                for i in 0..parts.len() {
                    if parts[i] == "cp" && i + 1 < parts.len() {
                        last_score = parts[i + 1].parse().unwrap_or(last_score);
                    } else if parts[i] == "mate" && i + 1 < parts.len() {
                        let m: i32 = parts[i + 1].parse().unwrap_or(0);
                        last_score = if m > 0 { 10000 - m } else { -10000 - m };
                    }
                }
            }

            if line.contains("bestmove") {
                break;
            }
        }
        Ok(last_score)
    }

    pub fn get_top_moves(
        &mut self,
        fen: &str,
        depth: u32,
        multipv: u32,
    ) -> Result<Vec<(String, i32)>> {
        self.is_ready()?;
        self.send_command(&format!("setoption name MultiPV value {}", multipv))?;
        self.is_ready()?;
        self.send_command(&format!("position fen {}", fen))?;
        self.send_command(&format!("go depth {}", depth))?;

        let mut moves = Vec::new();
        let stdout = self.process.stdout.as_mut().unwrap();
        let mut reader = BufReader::new(stdout);

        loop {
            let mut line = String::new();
            if reader.read_line(&mut line)? == 0 {
                break;
            }

            if line.contains("score cp") || line.contains("score mate") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                let mut cp = 0;
                let mut pv = String::new();
                for i in 0..parts.len() {
                    if parts[i] == "cp" && i + 1 < parts.len() {
                        cp = parts[i + 1].parse().unwrap_or(0);
                    } else if parts[i] == "mate" && i + 1 < parts.len() {
                        let m: i32 = parts[i + 1].parse().unwrap_or(0);
                        cp = if m > 0 { 10000 - m } else { -10000 - m };
                    }
                    if parts[i] == "pv" && i + 1 < parts.len() {
                        pv = parts[i + 1].to_string();
                    }
                }
                if !pv.is_empty() {
                    moves.push((pv, cp));
                }
            }

            if line.contains("bestmove") {
                break;
            }
        }

        // Reset MultiPV
        self.send_command("setoption name MultiPV value 1")?;
        self.is_ready()?;

        // MultiPV output usually gives 1..N. We only want the last batch of depth N.
        // The most reliable way is to take the last N unique PVs.
        moves.reverse();
        let mut seen = std::collections::HashSet::new();
        let mut final_moves = Vec::new();
        for (pv, cp) in moves {
            if seen.len() >= multipv as usize {
                break;
            }
            if !seen.contains(&pv) {
                seen.insert(pv.clone());
                final_moves.push((pv, cp));
            }
        }

        Ok(final_moves)
    }
}

pub struct ExplainableEngine {
    uci: UciEngine,
    pos: Chess,
    history: Vec<Move>,
    pub tb: Option<crate::syzygy::SyzygyTablebase>,
}

impl ExplainableEngine {
    pub fn new(stockfish_path: &str) -> Result<Self> {
        let uci = UciEngine::new(stockfish_path)?;
        Ok(ExplainableEngine {
            uci,
            pos: Chess::default(),
            history: Vec::new(),
            tb: None,
        })
    }

    pub fn make_move(&mut self, move_uci: &str) -> Result<()> {
        let uci_move: shakmaty::uci::UciMove = move_uci
            .parse()
            .map_err(|e| anyhow!("Invalid move format {}: {:?}", move_uci, e))?;

        let m = uci_move
            .to_move(&self.pos)
            .map_err(|e| anyhow!("Illegal or invalid move {}: {:?}", move_uci, e))?;

        self.pos.play_unchecked(m);
        self.history.push(m);
        Ok(())
    }

    pub fn set_position(&mut self, fen: &str) -> Result<()> {
        let setup: Fen = fen.parse().map_err(|e| anyhow!("Invalid FEN: {:?}", e))?;
        self.pos = setup
            .into_position(shakmaty::CastlingMode::Standard)
            .map_err(|e| anyhow!("Invalid position: {:?}", e))?;
        self.history.clear();
        Ok(())
    }

    pub fn get_best_move(&mut self, depth: u32) -> Result<String> {
        let fen = Fen::from_position(&self.pos, shakmaty::EnPassantMode::Always).to_string();
        self.uci.get_best_move(&fen, depth)
    }
}
