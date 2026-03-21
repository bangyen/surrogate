use shakmaty::{attacks, Bitboard, Color, Role, Square};

use crate::eval::piece_value;

/// Return the least-valuable attacker of `sq` belonging to `side`.
/// Uses the standard approach: try pawns first, then knights, bishops,
/// rooks, queens, king.  Returns `None` when `side` has no attacker.
pub fn least_valuable_attacker(
    board: &shakmaty::Board,
    sq: Square,
    side: Color,
    occupied: Bitboard,
) -> Option<(Square, Role)> {
    let by_side = board.by_color(side);
    // Pawns
    let pawn_attackers =
        attacks::pawn_attacks(side.other(), sq) & board.by_role(Role::Pawn) & by_side & occupied;
    if let Some(a) = pawn_attackers.into_iter().next() {
        return Some((a, Role::Pawn));
    }
    // Knights
    let knight_attackers =
        attacks::knight_attacks(sq) & board.by_role(Role::Knight) & by_side & occupied;
    if let Some(a) = knight_attackers.into_iter().next() {
        return Some((a, Role::Knight));
    }
    // Bishops
    let bishop_attackers =
        attacks::bishop_attacks(sq, occupied) & board.by_role(Role::Bishop) & by_side & occupied;
    if let Some(a) = bishop_attackers.into_iter().next() {
        return Some((a, Role::Bishop));
    }
    // Rooks
    let rook_attackers =
        attacks::rook_attacks(sq, occupied) & board.by_role(Role::Rook) & by_side & occupied;
    if let Some(a) = rook_attackers.into_iter().next() {
        return Some((a, Role::Rook));
    }
    // Queens
    let queen_attackers = (attacks::bishop_attacks(sq, occupied)
        | attacks::rook_attacks(sq, occupied))
        & board.by_role(Role::Queen)
        & by_side
        & occupied;
    if let Some(a) = queen_attackers.into_iter().next() {
        return Some((a, Role::Queen));
    }
    // King
    let king_attackers = attacks::king_attacks(sq) & board.by_role(Role::King) & by_side & occupied;
    if let Some(a) = king_attackers.into_iter().next() {
        return Some((a, Role::King));
    }
    None
}

/// Static Exchange Evaluation: determines the net material gain/loss
/// from a sequence of captures on `target` square, assuming both sides
/// always recapture with their least valuable attacker.
///
/// A positive SEE means the initial capture wins material; negative
/// means it loses material.  This gives far more accurate tactical
/// assessment than the binary "attacked and undefended" hanging check.
///
/// `attacker_sq` is the square of the piece initiating the capture.
pub fn see(board: &shakmaty::Board, target: Square, attacker_sq: Square) -> i32 {
    let attacker_piece = match board.piece_at(attacker_sq) {
        Some(p) => p,
        None => return 0,
    };
    let victim_piece = match board.piece_at(target) {
        Some(p) => p,
        None => return 0,
    };

    // Gain array: gain[i] is the material balance from the i-th capture's
    // perspective.  We fill it iteratively, then propagate backwards.
    let mut gain: [i32; 33] = [0; 33]; // max 32 pieces on board + 1
    let mut depth = 0;
    let mut side = attacker_piece.color;
    let mut occupied = board.occupied();

    gain[0] = piece_value(victim_piece.role);

    // Remove the initial attacker from the occupied set so x-ray
    // attacks through it are revealed.
    occupied ^= Bitboard::from_square(attacker_sq);

    let mut current_attacker_value = piece_value(attacker_piece.role);

    loop {
        depth += 1;
        side = side.other();

        // The gain for this depth is the value of the piece just captured
        // minus the gain that the opponent will realise in subsequent
        // captures (filled in later via the negamax-style propagation).
        gain[depth] = current_attacker_value - gain[depth - 1];

        // Pruning: if gain cannot improve the situation for the moving side,
        // stop early (stand-pat optimisation).
        if (-gain[depth - 1]).max(gain[depth]) < 0 {
            break;
        }

        if let Some((sq, role)) = least_valuable_attacker(board, target, side, occupied) {
            current_attacker_value = piece_value(role);
            occupied ^= Bitboard::from_square(sq);
        } else {
            break;
        }
    }

    // Propagate backwards: each side chooses max(not-capturing, capturing).
    while depth > 1 {
        depth -= 1;
        gain[depth - 1] = -((-gain[depth - 1]).max(gain[depth]));
    }

    gain[0]
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
    fn test_see_winning_capture() {
        // Pawn captures undefended knight: SEE should be positive (~220 cp).
        let pos = pos_from_fen("rnbqkb1r/pppppppp/8/4n3/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 1");
        let board = pos.board();
        let see_val = see(board, Square::E5, Square::D4);
        assert!(
            see_val > 0,
            "Pawn x undefended Knight should win: got {see_val}"
        );
    }

    #[test]
    fn test_see_losing_capture() {
        // Queen captures defended pawn: SEE should be negative.
        let pos = pos_from_fen("rnbqkbnr/ppp2ppp/3p4/4p3/8/8/PPPPQPPP/RNB1KBNR w KQkq - 0 1");
        let board = pos.board();
        let see_val = see(board, Square::E5, Square::E2);
        assert!(
            see_val < 0,
            "Queen x defended Pawn should lose: got {see_val}"
        );
    }

    #[test]
    fn test_see_equal_exchange() {
        // Knight captures knight (equal trade).
        let pos = pos_from_fen("r1bqkbnr/ppp1pppp/3p4/4n3/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1");
        let board = pos.board();
        let see_val = see(board, Square::E5, Square::F3);
        assert!(
            see_val.abs() <= 10,
            "Knight x Knight should be ~0: got {see_val}"
        );
    }
}
