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

/// Get the PST value for a piece on a square, correctly flipped for the side.
#[inline]
pub fn pst_value(role: Role, sq: Square, color: Color, phase_count: i32) -> i32 {
    let idx = pst_index(sq, color);
    let pf = phase_factor(phase_count);
    match role {
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
    }
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

    let mut score = 0i32;

    for sq in Square::ALL {
        if let Some(piece) = board.piece_at(sq) {
            let mat = piece_value(piece.role);
            let pst = pst_value(piece.role, sq, piece.color, phase);
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
    use shakmaty::fen::Fen;
    use shakmaty::{CastlingMode, Chess};

    fn pos_from_fen(fen: &str) -> Chess {
        let setup: Fen = fen.parse().unwrap();
        setup.into_position(CastlingMode::Standard).unwrap()
    }

    #[test]
    fn test_evaluate_symmetry() {
        // Evaluation should be symmetric: evaluate(pos) == -evaluate(flipped_pos).
        // Since evaluate() returns score from side-to-move's perspective,
        // if we flip the board and the turn, the score should be the same
        // if the position is functionally identical but mirrored.
        let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        let pos = pos_from_fen(fen);
        let score = evaluate(&pos);

        // Mirror the position: flip all pieces and the turn.
        // shakmaty doesn't have a built-in 'flip' but we can manually
        // construct a mirrored FEN or position.
        // For the starting position, it's already symmetric.
        assert_eq!(score, 0);

        // Test an asymmetric position
        let pos2 = pos_from_fen("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 1 2");
        let score2 = evaluate(&pos2);

        // Mirrored position for Black
        let pos2_mirrored =
            pos_from_fen("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 1 2");
        let score2_mirrored = evaluate(&pos2_mirrored);

        // From relative perspective, white's advantage in pos2 should equal black's in pos2_mirrored.
        assert_eq!(
            score2, score2_mirrored,
            "Symmetry failure: {} vs {}",
            score2, score2_mirrored
        );
    }

    #[test]
    fn test_pst_knight_center_vs_edge() {
        // Knight on e4 (center) vs Knight on a1 (edge). Need kings for valid position.
        let pos_center = pos_from_fen("k7/8/8/8/4N3/8/8/K7 w - - 0 1");
        let pos_edge = pos_from_fen("k7/8/8/8/8/8/8/N3K3 w - - 0 1");

        let score_center = evaluate(&pos_center);
        let score_edge = evaluate(&pos_edge);

        assert!(
            score_center > score_edge,
            "Knight in center ({}) should score higher than on edge ({})",
            score_center,
            score_edge
        );
    }

    #[test]
    fn test_pst_pawn_advancement() {
        // White pawn on e2 vs White pawn on e7. Need kings.
        let pos_e2 = pos_from_fen("k7/8/8/8/8/8/4P3/4K3 w - - 0 1");
        let pos_e7 = pos_from_fen("k7/4P3/8/8/8/8/8/4K3 w - - 0 1");

        let score_e2 = evaluate(&pos_e2);
        let score_e7 = evaluate(&pos_e7);

        assert!(
            score_e7 > score_e2,
            "Advanced pawn ({}) should score higher than starting pawn ({})",
            score_e7,
            score_e2
        );
    }

    #[test]
    fn test_bishop_pair_bonus() {
        // White has 2 bishops, Black has 1 knight + 1 bishop. Need kings.
        let pos_pair = pos_from_fen("k7/8/8/4BB2/8/8/8/4K3 w - - 0 1");
        let pos_no_pair = pos_from_fen("k7/8/8/4BN2/8/8/8/4K3 w - - 0 1");

        let score_pair = evaluate(&pos_pair);
        let score_no_pair = evaluate(&pos_no_pair);

        // Difference should be roughly 30 (pair bonus) + (bishop_val - knight_val)
        let diff = score_pair - score_no_pair;
        assert!(
            diff >= 30,
            "Bishop pair bonus should be evident: diff was {}",
            diff
        );
    }

    #[test]
    fn test_pst_comprehensive() {
        use shakmaty::{Role, Square};
        // Verify that PST penalties are actually penalties and bonuses are bonuses.
        let roles = [
            Role::Pawn,
            Role::Knight,
            Role::Bishop,
            Role::Rook,
            Role::Queen,
            Role::King,
        ];

        for role in roles {
            for square in Square::ALL {
                let pst = pst_value(role, square, shakmaty::Color::White, 14);
                if pst == 0 {
                    continue;
                }

                // We'll use a board with the piece at 'square' and kings at far corners.

                // Anchor points for catching sign-deletion in PST tables.
                if role == Role::Pawn && square == Square::D2 {
                    // Pawn on D2 has penalty -20
                    assert!(pst < 0, "Pawn on D2 should have a PST penalty: got {}", pst);
                }
                if role == Role::Knight && square == Square::A1 {
                    assert!(
                        pst < 0,
                        "Knight on A1 should have a PST penalty: got {}",
                        pst
                    );
                }
                if role == Role::Knight && square == Square::E4 {
                    assert!(pst > 0, "Knight on E4 should have a PST bonus: got {}", pst);
                }
                if role == Role::Bishop && square == Square::A1 {
                    assert!(
                        pst < 0,
                        "Bishop on A1 should have a PST penalty: got {}",
                        pst
                    );
                }
                if role == Role::Rook && square == Square::A1 {
                    assert!(pst == 0, "Rook on A1 should have 0 pst: got {}", pst);
                }
                if role == Role::Rook && square == Square::A7 {
                    // White Rook on 7th rank (A7) bonus is 5
                    assert!(pst > 0, "Rook on 7th rank bonus: got {}", pst);
                }
                if role == Role::Queen && square == Square::A1 {
                    assert!(
                        pst < 0,
                        "Queen on A1 should have a PST penalty: got {}",
                        pst
                    );
                }
                if role == Role::King && square == Square::G1 {
                    // Middlegame King at G1 (castled) is a bonus 30
                    assert!(
                        pst > 0,
                        "King on G1 (MG) should have a PST bonus: got {}",
                        pst
                    );
                }
            }
        }
    }

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
