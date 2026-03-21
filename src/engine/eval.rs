use shakmaty::{Chess, Color, Position, Role, Square};

/// Material values in centipawns, used throughout the crate for
/// piece valuation in evaluation, SEE, and move ordering.
pub fn piece_value(role: Role) -> i32 {
    match role {
        Role::Pawn => 100,
        Role::Knight => 320,
        Role::Bishop => 330,
        Role::Rook => 500,
        Role::Queen => 900,
        Role::King => 20000,
    }
}

// ── Piece-square tables (from White's perspective, rank-8 = index 0) ──

pub static EVAL_PST_PAWN: [i32; 64] = [
    0, 0, 0, 0, 0, 0, 0, 0, 50, 50, 50, 50, 50, 50, 50, 50, 10, 10, 20, 30, 30, 20, 10, 10, 5, 5,
    10, 25, 25, 10, 5, 5, 0, 0, 0, 20, 20, 0, 0, 0, 5, -5, -10, 0, 0, -10, -5, 5, 5, 10, 10, -20,
    -20, 10, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0,
];

pub static EVAL_PST_KNIGHT: [i32; 64] = [
    -50, -40, -30, -30, -30, -30, -40, -50, -40, -20, 0, 0, 0, 0, -20, -40, -30, 0, 10, 15, 15, 10,
    0, -30, -30, 5, 15, 20, 20, 15, 5, -30, -30, 0, 15, 20, 20, 15, 0, -30, -30, 5, 10, 15, 15, 10,
    5, -30, -40, -20, 0, 5, 5, 0, -20, -40, -50, -40, -30, -30, -30, -30, -40, -50,
];

pub static EVAL_PST_BISHOP: [i32; 64] = [
    -20, -10, -10, -10, -10, -10, -10, -20, -10, 0, 0, 0, 0, 0, 0, -10, -10, 0, 5, 10, 10, 5, 0,
    -10, -10, 5, 5, 10, 10, 5, 5, -10, -10, 0, 10, 10, 10, 10, 0, -10, -10, 10, 10, 10, 10, 10, 10,
    -10, -10, 5, 0, 0, 0, 0, 5, -10, -20, -10, -10, -10, -10, -10, -10, -20,
];

pub static EVAL_PST_ROOK: [i32; 64] = [
    0, 0, 0, 0, 0, 0, 0, 0, 5, 10, 10, 10, 10, 10, 10, 5, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0,
    0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, 0, 0,
    0, 5, 5, 0, 0, 0,
];

pub static EVAL_PST_QUEEN: [i32; 64] = [
    -20, -10, -10, -5, -5, -10, -10, -20, -10, 0, 0, 0, 0, 0, 0, -10, -10, 0, 5, 5, 5, 5, 0, -10,
    -5, 0, 5, 5, 5, 5, 0, -5, -5, 0, 5, 5, 5, 5, 0, -5, -10, 5, 5, 5, 5, 5, 0, -10, -10, 0, 5, 0,
    0, 0, 0, -10, -20, -10, -10, -5, -5, -10, -10, -20,
];

pub static EVAL_PST_KING_MG: [i32; 64] = [
    -30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40,
    -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30, -20, -30, -30, -40, -40, -30,
    -30, -20, -10, -20, -20, -20, -20, -20, -20, -10, 20, 20, 0, 0, 0, 0, 20, 20, 20, 30, 10, 0, 0,
    10, 30, 20,
];

pub static EVAL_PST_KING_EG: [i32; 64] = [
    -50, -40, -30, -20, -20, -30, -40, -50, -30, -20, -10, 0, 0, -10, -20, -30, -30, -10, 20, 30,
    30, 20, -10, -30, -30, -10, 30, 40, 40, 30, -10, -30, -30, -10, 30, 40, 40, 30, -10, -30, -30,
    -10, 20, 30, 30, 20, -10, -30, -30, -30, 0, 0, 0, 0, -30, -30, -50, -30, -30, -30, -30, -30,
    -30, -50,
];

/// Count non-pawn, non-king pieces for game-phase detection.
pub fn count_phase(board: &shakmaty::Board) -> i32 {
    let mut phase = 0i32;
    for sq in Square::ALL {
        if let Some(piece) = board.piece_at(sq) {
            if piece.role != Role::Pawn && piece.role != Role::King {
                phase += 1;
            }
        }
    }
    phase
}

/// PST index for a square, flipped for Black so tables are always
/// written from White's perspective.
#[inline]
pub fn pst_index(sq: Square, color: Color) -> usize {
    let vis_r = if color == Color::White {
        7 - sq.rank() as usize
    } else {
        sq.rank() as usize
    };
    vis_r * 8 + sq.file() as usize
}

/// Compute a continuous game-phase factor in 0.0..=1.0 where 0.0 is
/// pure endgame and 1.0 is the opening.  Based on total non-pawn,
/// non-king piece counts: max count of 14 maps to 1.0.
#[inline]
pub fn phase_factor(phase_count: i32) -> f32 {
    (phase_count as f32 / 14.0).clamp(0.0, 1.0)
}

/// Positional evaluation: material + phase-interpolated piece-square
/// tables + bishop-pair bonus.  Returns score from the side-to-move's
/// perspective so negamax works directly.
///
/// The king PST smoothly blends between middlegame and endgame values
/// using a continuous phase factor instead of a binary threshold,
/// giving more accurate positional scores in transitional positions.
pub fn evaluate(pos: &Chess) -> i32 {
    let board = pos.board();
    let phase = count_phase(board);
    let pf = phase_factor(phase);

    let mut score = 0i32;

    for sq in Square::ALL {
        if let Some(piece) = board.piece_at(sq) {
            let mat = piece_value(piece.role);
            let idx = pst_index(sq, piece.color);
            let pst = match piece.role {
                Role::Pawn => EVAL_PST_PAWN[idx],
                Role::Knight => EVAL_PST_KNIGHT[idx],
                Role::Bishop => EVAL_PST_BISHOP[idx],
                Role::Rook => EVAL_PST_ROOK[idx],
                Role::Queen => EVAL_PST_QUEEN[idx],
                Role::King => {
                    let mg = EVAL_PST_KING_MG[idx] as f32;
                    let eg = EVAL_PST_KING_EG[idx] as f32;
                    (pf * mg + (1.0 - pf) * eg) as i32
                }
            };
            let val = mat + pst;
            if piece.color == Color::White {
                score += val;
            } else {
                score -= val;
            }
        }
    }

    // Bishop-pair bonus (+30 cp)
    let white_bishops = (board.by_role(Role::Bishop) & board.by_color(Color::White)).count();
    let black_bishops = (board.by_role(Role::Bishop) & board.by_color(Color::Black)).count();
    if white_bishops >= 2 {
        score += 30;
    }
    if black_bishops >= 2 {
        score -= 30;
    }

    if pos.turn() == Color::White {
        score
    } else {
        -score
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_factor_opening() {
        // Full complement: 14 non-pawn, non-king pieces => factor = 1.0.
        assert!((phase_factor(14) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_phase_factor_endgame() {
        // No pieces left => factor = 0.0 (pure endgame).
        assert!(phase_factor(0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_phase_factor_midgame() {
        // 7 pieces => factor = 0.5.
        assert!((phase_factor(7) - 0.5).abs() < f32::EPSILON);
    }
}
