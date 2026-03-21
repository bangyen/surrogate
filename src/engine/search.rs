use crate::eval::{evaluate, piece_value};
use crate::zobrist::{piece_index, zobrist_hash};
use shakmaty::fen::Fen;
use shakmaty::{CastlingMode, Chess, Move, Position, Role};

// ── Transposition Table & Search Context ─────────────────────────────

pub const MAX_PLY: usize = 64;
pub const TT_SIZE: usize = 1 << 20;

/// Whether a transposition-table score is exact, a lower bound
/// (failed high), or an upper bound (failed low).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TTFlag {
    Exact,
    LowerBound,
    UpperBound,
}

/// A single entry in the transposition table, storing the best
/// information discovered for one position at a given search depth.
#[derive(Clone)]
pub struct TTEntry {
    pub key: u64,
    pub depth: u8,
    pub score: i32,
    pub flag: TTFlag,
    pub best_move: Option<Move>,
}

impl Default for TTEntry {
    fn default() -> Self {
        TTEntry {
            key: 0,
            depth: 0,
            score: 0,
            flag: TTFlag::Exact,
            best_move: None,
        }
    }
}

/// Mutable state threaded through the search: transposition table,
/// killer-move slots, and history heuristic counters.
pub struct SearchContext {
    pub tt: Vec<TTEntry>,
    pub killers: [[Option<Move>; 2]; MAX_PLY],
    pub history: [[i32; 64]; 12],
    pub ply: usize,
    pub allow_null: bool,
}

impl Default for SearchContext {
    /// Allocate a fresh search context with an empty TT.
    fn default() -> Self {
        SearchContext {
            tt: vec![TTEntry::default(); TT_SIZE],
            killers: [[None; 2]; MAX_PLY],
            history: [[0; 64]; 12],
            ply: 0,
            allow_null: true,
        }
    }
}

impl SearchContext {
    /// Allocate a fresh search context with an empty TT.
    pub fn new() -> Self {
        Self::default()
    }

    /// Probe the TT for a matching entry.  Returns `None` when the
    /// slot is empty or holds a different position.
    pub fn tt_probe(&self, key: u64) -> Option<&TTEntry> {
        let idx = (key as usize) % TT_SIZE;
        let entry = &self.tt[idx];
        if entry.key == key {
            Some(entry)
        } else {
            None
        }
    }

    /// Store search results in the TT (always-replace policy).
    pub fn tt_store(
        &mut self,
        key: u64,
        depth: u8,
        score: i32,
        flag: TTFlag,
        best_move: Option<Move>,
    ) {
        let idx = (key as usize) % TT_SIZE;
        self.tt[idx] = TTEntry {
            key,
            depth,
            score,
            flag,
            best_move,
        };
    }

    /// Record a killer move for the current ply (quiet moves that
    /// caused a beta cutoff).
    pub fn store_killer(&mut self, ply: usize, m: Move) {
        if ply < MAX_PLY && self.killers[ply][0] != Some(m) {
            self.killers[ply][1] = self.killers[ply][0];
            self.killers[ply][0] = Some(m);
        }
    }

    /// Bump the history counter for a quiet move that caused a beta
    /// cutoff, aging all counters when they grow too large.
    pub fn update_history(&mut self, pos: &Chess, m: &Move, depth: u8) {
        if let Some(from_sq) = m.from() {
            if let Some(piece) = pos.board().piece_at(from_sq) {
                let pi = piece_index(piece.color, piece.role);
                let to = m.to() as usize;
                self.history[pi][to] += (depth as i32) * (depth as i32);
                if self.history[pi][to] > 1_000_000 {
                    for p in self.history.iter_mut() {
                        for s in p.iter_mut() {
                            *s /= 2;
                        }
                    }
                }
            }
        }
    }
}

/// Assign a priority score to each legal move so the most promising
/// moves are searched first.  Ordering: TT move > promotions >
/// captures (MVV-LVA) > killers > history heuristic.
fn score_moves(
    pos: &Chess,
    moves: &[Move],
    ctx: &SearchContext,
    tt_move: Option<&Move>,
) -> Vec<(i32, Move)> {
    moves
        .iter()
        .map(|m| {
            let score;
            if tt_move == Some(m) {
                score = 30000;
            } else if m.is_promotion() {
                score = 20000;
            } else if m.is_capture() {
                let board = pos.board();
                let victim = board.piece_at(m.to()).map(|p| p.role).unwrap_or(Role::Pawn);
                let attacker = m
                    .from()
                    .and_then(|sq| board.piece_at(sq))
                    .map(|p| p.role)
                    .unwrap_or(Role::Pawn);
                score = 10000 + piece_value(victim) - piece_value(attacker);
            } else if ctx.ply < MAX_PLY && ctx.killers[ctx.ply][0] == Some(*m) {
                score = 9000;
            } else if ctx.ply < MAX_PLY && ctx.killers[ctx.ply][1] == Some(*m) {
                score = 8999;
            } else {
                score = m
                    .from()
                    .and_then(|sq| pos.board().piece_at(sq))
                    .map(|p| {
                        let pi = piece_index(p.color, p.role);
                        ctx.history[pi][m.to() as usize].min(8998)
                    })
                    .unwrap_or(0);
            }
            (score, *m)
        })
        .collect()
}

/// Quiescence search with delta pruning: keeps searching tactical
/// moves (captures / promotions) until the position is quiet, while
/// pruning captures that cannot possibly raise alpha.
fn quiesce(pos: &Chess, mut alpha: i32, beta: i32) -> i32 {
    let stand_pat = evaluate(pos);
    if stand_pat >= beta {
        return beta;
    }
    if stand_pat > alpha {
        alpha = stand_pat;
    }

    let moves = pos.legal_moves();
    let mut tactical: Vec<(i32, Move)> = moves
        .into_iter()
        .filter(|m| m.is_capture() || m.is_promotion())
        .map(|m| {
            let mut score = 0i32;
            if m.is_capture() {
                let victim = pos
                    .board()
                    .piece_at(m.to())
                    .map(|p| p.role)
                    .unwrap_or(Role::Pawn);
                let attacker = pos
                    .board()
                    .piece_at(m.from().unwrap())
                    .map(|p| p.role)
                    .unwrap_or(Role::Pawn);
                score = 10000 + piece_value(victim) - piece_value(attacker);
            }
            if m.is_promotion() {
                score += 20000;
            }
            (score, m)
        })
        .collect();
    tactical.sort_by(|a, b| b.0.cmp(&a.0));

    for (_, m) in tactical {
        // Delta pruning: skip captures whose best-case gain cannot
        // reach alpha (promotions are always searched).
        if !m.is_promotion() && m.is_capture() {
            let victim_val = pos
                .board()
                .piece_at(m.to())
                .map(|p| piece_value(p.role))
                .unwrap_or(100);
            if stand_pat + victim_val + 200 < alpha {
                continue;
            }
        }

        let mut new_pos = pos.clone();
        new_pos.play_unchecked(m);
        let score = -quiesce(&new_pos, -beta, -alpha);
        if score >= beta {
            return beta;
        }
        if score > alpha {
            alpha = score;
        }
    }
    alpha
}

/// Alpha-beta search enhanced with transposition table lookups,
/// null-move pruning, and late move reductions.  Falls back to
/// quiescence search at the leaves.
fn alpha_beta(pos: &Chess, mut alpha: i32, beta: i32, depth: u8, ctx: &mut SearchContext) -> i32 {
    // ── Terminal checks ──────────────────────────────────────────────
    if pos.is_game_over() {
        if pos.is_checkmate() {
            return -30000 + ctx.ply as i32;
        }
        return 0;
    }
    if depth == 0 {
        return quiesce(pos, alpha, beta);
    }

    let hash = zobrist_hash(pos);
    let orig_alpha = alpha;

    // ── TT probe ─────────────────────────────────────────────────────
    let tt_move: Option<Move>;
    if let Some(entry) = ctx.tt_probe(hash) {
        tt_move = entry.best_move;
        if entry.depth >= depth {
            match entry.flag {
                TTFlag::Exact => return entry.score,
                TTFlag::LowerBound => {
                    if entry.score >= beta {
                        return entry.score;
                    }
                }
                TTFlag::UpperBound => {
                    if entry.score <= alpha {
                        return entry.score;
                    }
                }
            }
        }
    } else {
        tt_move = None;
    }

    let in_check = pos.is_check();

    // ── Null-move pruning ────────────────────────────────────────────
    if ctx.allow_null && !in_check && depth >= 3 {
        let board = pos.board();
        let us = pos.turn();
        let has_pieces =
            !(board.by_color(us) & !board.by_role(Role::Pawn) & !board.by_role(Role::King))
                .is_empty();
        if has_pieces {
            if let Ok(null_pos) = pos.clone().swap_turn() {
                let r: u8 = if depth >= 6 { 3 } else { 2 };
                let prev_null = ctx.allow_null;
                ctx.allow_null = false;
                ctx.ply += 1;
                let null_score = -alpha_beta(&null_pos, -beta, -beta + 1, depth - 1 - r, ctx);
                ctx.ply -= 1;
                ctx.allow_null = prev_null;
                if null_score >= beta {
                    return beta;
                }
            }
        }
    }

    // ── Move generation & ordering ───────────────────────────────────
    let moves = pos.legal_moves();
    if moves.is_empty() {
        return if in_check { -30000 + ctx.ply as i32 } else { 0 };
    }

    let move_list: Vec<Move> = moves.into_iter().collect();
    let mut scored = score_moves(pos, &move_list, ctx, tt_move.as_ref());
    scored.sort_by(|a, b| b.0.cmp(&a.0));

    let mut best_score = -50000i32;
    let mut best_move: Option<Move> = None;
    let ply = ctx.ply;

    for (i, (_, m)) in scored.iter().enumerate() {
        let mut new_pos = pos.clone();
        new_pos.play_unchecked(*m);
        let gives_check = new_pos.is_check();

        // ── Late Move Reductions (LMR) ──────────────────────────────
        let score;
        if i >= 4 && depth >= 3 && !m.is_capture() && !m.is_promotion() && !gives_check && !in_check
        {
            let reduction: u8 = if depth >= 6 && i >= 8 { 2 } else { 1 };
            let reduced_depth = depth - 1 - reduction;
            ctx.ply += 1;
            let lmr_score = -alpha_beta(&new_pos, -alpha - 1, -alpha, reduced_depth, ctx);
            ctx.ply -= 1;

            if lmr_score > alpha {
                // Re-search at full depth to verify.
                ctx.ply += 1;
                score = -alpha_beta(&new_pos, -beta, -alpha, depth - 1, ctx);
                ctx.ply -= 1;
            } else {
                score = lmr_score;
            }
        } else {
            ctx.ply += 1;
            score = -alpha_beta(&new_pos, -beta, -alpha, depth - 1, ctx);
            ctx.ply -= 1;
        }

        if score > best_score {
            best_score = score;
            best_move = Some(*m);
        }
        if score >= beta {
            if !m.is_capture() && !m.is_promotion() {
                ctx.store_killer(ply, *m);
                ctx.update_history(pos, m, depth);
            }
            ctx.tt_store(hash, depth, beta, TTFlag::LowerBound, Some(*m));
            return beta;
        }
        if score > alpha {
            alpha = score;
        }
    }

    let flag = if best_score <= orig_alpha {
        TTFlag::UpperBound
    } else {
        TTFlag::Exact
    };
    ctx.tt_store(hash, depth, best_score, flag, best_move);
    best_score
}

/// Core iterative-deepening search with aspiration windows.  The
/// transposition table persists across iterations so each deeper
/// search benefits from earlier results.
pub fn find_best_reply_impl(pos: &Chess, depth: u8) -> Option<String> {
    let moves = pos.legal_moves();
    if moves.is_empty() {
        return None;
    }

    let mut ctx = SearchContext::new();
    let mut best_move: Option<String> = None;
    let mut prev_score = 0i32;

    for d in 1..=depth {
        // Aspiration windows: narrow search around the previous score
        // for d >= 2, falling back to a full window on fail.
        let score;
        if d <= 1 {
            score = alpha_beta(pos, -50000, 50000, d, &mut ctx);
        } else {
            let a = prev_score - 25;
            let b = prev_score + 25;
            let s = alpha_beta(pos, a, b, d, &mut ctx);
            if s <= a || s >= b {
                // Window failed -- re-search with full bounds.
                score = alpha_beta(pos, -50000, 50000, d, &mut ctx);
            } else {
                score = s;
            }
        }
        prev_score = score;

        // Retrieve the best move for the root position from the TT.
        let hash = zobrist_hash(pos);
        if let Some(entry) = ctx.tt_probe(hash) {
            if let Some(m) = entry.best_move {
                best_move = Some(m.to_uci(CastlingMode::Standard).to_string());
            }
        }
    }

    // Fallback: return the first legal move if TT lookup failed.
    if best_move.is_none() {
        let first = pos.legal_moves().into_iter().next();
        if let Some(m) = first {
            best_move = Some(m.to_uci(CastlingMode::Standard).to_string());
        }
    }

    best_move
}

/// Find the best reply for the given FEN string at the given depth.
pub fn find_best_reply(fen: &str, depth: u8) -> anyhow::Result<Option<String>> {
    let setup: Fen = fen.parse().map_err(|_| anyhow::anyhow!("Invalid FEN"))?;
    let pos: Chess = setup
        .into_position(CastlingMode::Standard)
        .map_err(|_| anyhow::anyhow!("Invalid Position"))?;
    Ok(find_best_reply_impl(&pos, depth))
}

/// Core forcing-swing calculation: evaluates the largest evaluation
/// swing obtainable from a forcing move (capture or check).
pub fn calculate_forcing_swing_impl(pos: &Chess, depth: u8) -> f32 {
    let mut ctx = SearchContext::new();
    let base_eval = alpha_beta(pos, -50000, 50000, depth, &mut ctx);

    let moves = pos.legal_moves();
    let mut max_swing: f32 = 0.0;

    for m in moves {
        let is_capture = m.is_capture();
        let gives_check = {
            let mut test_pos = pos.clone();
            test_pos.play_unchecked(m);
            test_pos.is_check()
        };

        if is_capture || gives_check {
            let mut new_pos = pos.clone();
            new_pos.play_unchecked(m);
            let ev_after = alpha_beta(&new_pos, -50000, 50000, depth.saturating_sub(1), &mut ctx);
            let score_for_us = -ev_after;
            let swing = (score_for_us - base_eval) as f32;
            if swing > max_swing {
                max_swing = swing;
            }
        }
    }

    max_swing
}

/// Calculate the forcing swing for the given FEN string at the given depth.
pub fn calculate_forcing_swing(fen: &str, depth: u8) -> anyhow::Result<f32> {
    let setup: Fen = fen.parse().map_err(|_| anyhow::anyhow!("Invalid FEN"))?;
    let pos: Chess = setup
        .into_position(CastlingMode::Standard)
        .map_err(|_| anyhow::anyhow!("Invalid Position"))?;
    Ok(calculate_forcing_swing_impl(&pos, depth))
}

#[cfg(test)]
mod tests {
    use super::*;
    use shakmaty::fen::Fen;
    use shakmaty::{CastlingMode, Chess, Position};

    fn pos_from_fen(fen: &str) -> Chess {
        let setup: Fen = fen.parse().unwrap();
        setup.into_position(CastlingMode::Standard).unwrap()
    }

    #[test]
    fn test_tt_store_and_retrieve() {
        let mut ctx = SearchContext::new();
        let pos = pos_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let hash = zobrist_hash(&pos);

        ctx.tt_store(hash, 5, 42, TTFlag::Exact, None);

        let entry = ctx.tt_probe(hash).unwrap();
        assert_eq!(entry.depth, 5);
        assert_eq!(entry.score, 42);
        assert_eq!(entry.flag, TTFlag::Exact);
    }

    #[test]
    fn test_find_best_reply_mate_in_one() {
        let pos =
            pos_from_fen("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4");
        let result = find_best_reply_impl(&pos, 4);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), "h5f7");
    }

    #[test]
    fn test_find_best_reply_obvious_capture() {
        let pos = pos_from_fen("rnb1kbnr/pppppppp/8/4q3/4Q3/8/PPPP1PPP/RNB1KBNR w KQkq - 0 1");
        let result = find_best_reply_impl(&pos, 4);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), "e4e5");
    }

    #[test]
    fn test_quiesce_reasonable_score() {
        let pos = pos_from_fen("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2");
        let score = quiesce(&pos, -50000, 50000);
        assert!(
            score > -1000 && score < 1000,
            "Unexpected quiesce score: {score}"
        );
    }

    #[test]
    fn test_aspiration_returns_legal_move() {
        let pos = pos_from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1");
        let result = find_best_reply_impl(&pos, 5);
        assert!(result.is_some());
        let legal: Vec<String> = pos
            .legal_moves()
            .into_iter()
            .map(|m| m.to_uci(CastlingMode::Standard).to_string())
            .collect();
        assert!(legal.contains(&result.unwrap()));
    }

    #[test]
    fn test_depth_returns_valid_moves() {
        let pos = pos_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        for d in 1..=6u8 {
            let result = find_best_reply_impl(&pos, d);
            assert!(result.is_some(), "No move at depth {d}");
            let legal: Vec<String> = pos
                .legal_moves()
                .into_iter()
                .map(|m| m.to_uci(CastlingMode::Standard).to_string())
                .collect();
            assert!(
                legal.contains(&result.clone().unwrap()),
                "Illegal move {} at depth {d}",
                result.unwrap()
            );
        }
    }
}
