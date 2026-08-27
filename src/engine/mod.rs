pub mod eval;
pub mod search;
pub mod see;
pub mod zobrist;

use anyhow::{anyhow, Result};
use shakmaty::{fen::Fen, Chess, Move, Position};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::Mutex;
use std::thread;
use std::time::Duration;

/// Default time to wait for a single line of engine output.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// Score assigned to a forced mate, before the distance-to-mate
/// adjustment that makes shorter mates preferable.
const MATE_SCORE: i32 = 10000;

/// Parse the `score cp` / `score mate` field out of a UCI `info` line.
///
/// Mate scores are folded into the centipawn scale so callers can treat
/// every score uniformly.  Returns `None` when the line carries no score.
fn parse_score(parts: &[&str]) -> Option<i32> {
    let mut score = None;
    for i in 0..parts.len() {
        if parts[i] == "cp" && i + 1 < parts.len() {
            if let Ok(cp) = parts[i + 1].parse::<i32>() {
                score = Some(cp);
            }
        } else if parts[i] == "mate" && i + 1 < parts.len() {
            if let Ok(m) = parts[i + 1].parse::<i32>() {
                score = Some(if m > 0 {
                    MATE_SCORE - m
                } else {
                    -MATE_SCORE - m
                });
            }
        }
    }
    score
}

pub struct UciEngine {
    process: Child,
    /// Lines pumped off the engine's stdout by a background reader.
    ///
    /// A dedicated thread owns the `BufReader` for the life of the
    /// process.  Rebuilding a reader per call would discard whatever the
    /// previous one had buffered past the line we stopped at, silently
    /// losing engine output between commands.  Reading through a channel
    /// also lets us apply a real timeout, which a blocking `read_line`
    /// on a pipe cannot support.
    ///
    /// The `Mutex` is what makes the engine `Sync`, so callers can share
    /// one behind an `Arc<RwLock<_>>`; it is never actually contended,
    /// since every read goes through `&mut self`.
    lines: Mutex<Receiver<String>>,
}

impl UciEngine {
    pub fn new(path: &str) -> Result<Self> {
        let mut process = Command::new(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()?;

        let stdout = process
            .stdout
            .take()
            .ok_or_else(|| anyhow!("Failed to open stdout"))?;

        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                match line {
                    // A send error means the engine was dropped; stop.
                    Ok(l) => {
                        if tx.send(l).is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        });

        let mut engine = UciEngine {
            process,
            lines: Mutex::new(rx),
        };
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

    /// Read one line of engine output, giving up after `timeout`.
    fn read_line(&mut self, timeout: Duration, expected: &str) -> Result<String> {
        let lines = self
            .lines
            .get_mut()
            .map_err(|_| anyhow!("Engine reader lock poisoned"))?;
        match lines.recv_timeout(timeout) {
            Ok(line) => Ok(line),
            Err(RecvTimeoutError::Timeout) => Err(anyhow!(
                "Timed out after {:?} waiting for {}",
                timeout,
                expected
            )),
            Err(RecvTimeoutError::Disconnected) => Err(anyhow!(
                "Engine closed the connection waiting for {}",
                expected
            )),
        }
    }

    /// Block until a line containing `expected` arrives, or `timeout`
    /// elapses without the engine producing one.
    pub fn wait_for_line(&mut self, expected: &str, timeout: Duration) -> Result<String> {
        loop {
            let line = self.read_line(timeout, expected)?;
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

        loop {
            let line = self.read_line(DEFAULT_TIMEOUT, "bestmove")?;

            if line.contains("score cp") || line.contains("score mate") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if let Some(score) = parse_score(&parts) {
                    last_score = score;
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

        loop {
            let line = self.read_line(DEFAULT_TIMEOUT, "bestmove")?;

            if line.contains("score cp") || line.contains("score mate") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                let cp = parse_score(&parts).unwrap_or(0);
                let pv = parts
                    .iter()
                    .position(|p| *p == "pv")
                    .and_then(|i| parts.get(i + 1))
                    .map(|m| m.to_string())
                    .unwrap_or_default();

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

/// Shut the engine down when the handle goes away.
///
/// Without this, every `UciEngine` created over a process's lifetime
/// leaves an orphaned Stockfish behind — each one holding a parked
/// reader thread.  The long-running web server creates engines on
/// demand, so the leak accumulates there.
impl Drop for UciEngine {
    fn drop(&mut self) {
        // Ask politely first so the engine can exit on its own terms.
        // Dropping stdin also signals EOF, which most engines honour.
        let _ = self.send_command("quit");
        drop(self.process.stdin.take());

        // Give it a moment, then insist.  `try_wait` avoids blocking
        // forever on an engine that ignores `quit`.
        for _ in 0..50 {
            match self.process.try_wait() {
                Ok(Some(_)) => return,
                Ok(None) => thread::sleep(Duration::from_millis(10)),
                // Already reaped or unwaitable; nothing left to clean up.
                Err(_) => return,
            }
        }

        let _ = self.process.kill();
        let _ = self.process.wait();
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

#[cfg(test)]
mod tests {
    use super::*;

    fn parts(line: &str) -> Vec<&str> {
        line.split_whitespace().collect()
    }

    #[test]
    fn test_parse_score_centipawns() {
        let p = parts("info depth 10 score cp -37 nodes 1000 pv e2e4");
        assert_eq!(parse_score(&p), Some(-37));
    }

    #[test]
    fn test_parse_score_mate_prefers_shorter_mates() {
        let win_in_1 = parse_score(&parts("info score mate 1 pv h5f7")).unwrap();
        let win_in_5 = parse_score(&parts("info score mate 5 pv h5f7")).unwrap();
        assert!(
            win_in_1 > win_in_5,
            "mate in 1 ({win_in_1}) should beat mate in 5 ({win_in_5})"
        );
        assert!(win_in_5 > 0, "a mate we deliver must score positive");
    }

    #[test]
    fn test_parse_score_mate_against_us_is_negative() {
        let loss = parse_score(&parts("info score mate -3 pv e1e2")).unwrap();
        assert!(loss < 0, "being mated must score negative, got {loss}");
        // Being mated later is preferable to being mated sooner.
        let sooner = parse_score(&parts("info score mate -1 pv e1e2")).unwrap();
        assert!(
            loss > sooner,
            "mate in -3 ({loss}) should beat mate in -1 ({sooner})"
        );
    }

    #[test]
    fn test_parse_score_absent_or_malformed() {
        assert_eq!(parse_score(&parts("info depth 10 nodes 1000")), None);
        // A truncated line must not panic or misparse.
        assert_eq!(parse_score(&parts("info score cp")), None);
        assert_eq!(parse_score(&parts("info score mate")), None);
        assert_eq!(parse_score(&parts("info score cp abc")), None);
    }

    // ── Integration tests (require Stockfish) ────────────────────────

    /// Start an engine, or return `None` when Stockfish is unavailable so
    /// the suite still runs on machines without it.
    fn engine() -> Option<UciEngine> {
        let path = std::env::var("STOCKFISH_PATH").unwrap_or_else(|_| "stockfish".to_string());
        UciEngine::new(&path).ok()
    }

    #[test]
    fn test_repeated_calls_do_not_lose_buffered_output() {
        let Some(mut e) = engine() else {
            eprintln!("skipping: Stockfish not available");
            return;
        };
        let start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

        // Each call must leave the stream in a usable state for the next
        // one.  These are regression guards for the shared reader: a
        // reader rebuilt per call destroys whatever it had buffered past
        // the line it stopped on, which desynchronises later commands.
        for i in 0..3 {
            let mv = e.get_best_move(start, 6).unwrap();
            assert_eq!(mv.len(), 4, "call {i} returned a malformed move: {mv}");

            let score = e.get_evaluation(start, 6).unwrap();
            assert!(
                score.abs() < 200,
                "call {i}: start position should be near equal, got {score}"
            );

            let top = e.get_top_moves(start, 6, 3).unwrap();
            assert_eq!(top.len(), 3, "call {i} did not return 3 MultiPV lines");
            for (pv, _) in &top {
                assert_eq!(pv.len(), 4, "call {i} returned a malformed PV move: {pv}");
            }
        }
    }

    #[test]
    fn test_mixed_command_sequence_stays_in_sync() {
        let Some(mut e) = engine() else {
            eprintln!("skipping: Stockfish not available");
            return;
        };
        // Interleaving MultiPV and single-PV queries across different
        // positions is what desynchronises a per-call reader.
        let mated_soon = "6k1/5ppp/8/8/8/8/8/R3R1K1 w - - 0 1";
        let start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

        let top = e.get_top_moves(mated_soon, 8, 2).unwrap();
        assert_eq!(top.len(), 2);

        // White is completely winning here.
        let score = e.get_evaluation(mated_soon, 8).unwrap();
        assert!(score > 300, "White should be winning, got {score}");

        // Switching back to a balanced position must give a fresh answer,
        // not a stale score left over in a buffer.
        let score = e.get_evaluation(start, 8).unwrap();
        assert!(score.abs() < 200, "start should be near equal, got {score}");
    }

    #[test]
    fn test_timeout_is_enforced_on_a_silent_engine() {
        // `cat` speaks no UCI, so it never answers - the read must time
        // out rather than block forever.
        let Ok(mut process) = Command::new("cat")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
        else {
            eprintln!("skipping: could not spawn cat");
            return;
        };
        let stdout = process.stdout.take().unwrap();
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                match line {
                    Ok(l) => {
                        if tx.send(l).is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        });
        let mut e = UciEngine {
            process,
            lines: Mutex::new(rx),
        };

        let start = std::time::Instant::now();
        let err = e
            .wait_for_line("uciok", Duration::from_millis(300))
            .expect_err("a silent engine must not satisfy wait_for_line");
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "wait_for_line blocked for {:?} instead of timing out",
            start.elapsed()
        );
        assert!(
            err.to_string().contains("Timed out"),
            "unexpected error: {err}"
        );
        // `Drop` cleans the child up; no manual kill needed.
    }

    #[test]
    fn test_drop_reaps_the_engine_process() {
        let Some(e) = engine() else {
            eprintln!("skipping: Stockfish not available");
            return;
        };
        let pid = e.process.id();
        drop(e);

        // A reaped child leaves no live process behind.  `kill -0` on a
        // still-running pid succeeds, so it must now fail.
        let alive = Command::new("kill")
            .args(["-0", &pid.to_string()])
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        assert!(!alive, "engine process {pid} survived the drop");
    }
}
