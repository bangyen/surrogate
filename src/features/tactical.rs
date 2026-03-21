use shakmaty::{Bitboard, Chess, Color, Position, Role};
use std::collections::BTreeMap;
use crate::eval::piece_value;

pub fn extract(pos: &Chess, feats: &mut BTreeMap<String, f32>, turn: Color, opp: Color) {
    let board = pos.board();
    let occupied = board.occupied();

    // Hanging Pieces
    let count_hanging = |side: Color| {
        let mut count = 0;
        let my_pieces = board.by_color(side);
        for sq in my_pieces {
            let attacked = board.attacks_to(sq, side.other(), occupied);
            let defended = board.attacks_to(sq, side, occupied);
            if !attacked.is_empty() && defended.is_empty() {
                count += 1;
            }
        }
        count as f32
    };
    feats.insert("hanging_us".to_string(), count_hanging(turn));
    feats.insert("hanging_them".to_string(), count_hanging(opp));

    // Bishop Pair
    let has_bishop_pair = |side: Color| {
        if (board.by_role(Role::Bishop) & board.by_color(side)).count() >= 2 {
            1.0
        } else {
            0.0
        }
    };
    feats.insert("bishop_pair_us".to_string(), has_bishop_pair(turn));
    feats.insert("bishop_pair_them".to_string(), has_bishop_pair(opp));

    // Rook on 7th
    let count_rook_7th = |side: Color| {
        let target_rank = match side {
            Color::White => shakmaty::Rank::Seventh,
            Color::Black => shakmaty::Rank::Second,
        };
        (board.by_role(Role::Rook) & board.by_color(side) & Bitboard::from_rank(target_rank))
            .count() as f32
    };
    feats.insert("rook_on_7th_us".to_string(), count_rook_7th(turn));
    feats.insert("rook_on_7th_them".to_string(), count_rook_7th(opp));

    // Outposts
    let count_outposts = |side: Color| {
        let mut count = 0;
        let knights = board.by_role(Role::Knight) & board.by_color(side);
        for sq in knights {
            let rank: shakmaty::Rank = sq.rank();
            let rel_rank = if side == Color::White { rank as usize } else { 7 - rank as usize };
            if !(3..=5).contains(&rel_rank) { continue; }

            let mut is_supported = false;
            for attacker_sq in board.attacks_to(sq, side, occupied) {
                if let Some(p) = board.piece_at(attacker_sq) {
                    if p.role == Role::Pawn { is_supported = true; break; }
                }
            }
            if !is_supported { continue; }

            let mut attacked_by_pawn = false;
            for attacker_sq in board.attacks_to(sq, side.other(), occupied) {
                if let Some(p) = board.piece_at(attacker_sq) {
                    if p.role == Role::Pawn { attacked_by_pawn = true; break; }
                }
            }
            if attacked_by_pawn { continue; }
            count += 1;
        }
        count as f32
    };
    feats.insert("outposts_us".to_string(), count_outposts(turn));
    feats.insert("outposts_them".to_string(), count_outposts(opp));

    // Pinned Pieces
    let count_pinned = |side: Color| {
        let mut count = 0;
        if let Some(king) = board.king_of(side) {
            let enemy_side = side.other();
            let snipers = (shakmaty::attacks::rook_attacks(king, Bitboard::EMPTY) & board.rooks_and_queens())
                | (shakmaty::attacks::bishop_attacks(king, Bitboard::EMPTY) & board.bishops_and_queens());

            let mut blockers = Bitboard::EMPTY;
            for sniper in snipers & board.by_color(enemy_side) {
                let b = shakmaty::attacks::between(king, sniper) & board.occupied();
                if !b.more_than_one() && !b.is_empty() {
                    blockers |= b;
                }
            }
            count = (blockers & board.by_color(side)).count();
        }
        count as f32
    };
    feats.insert("pinned_us".to_string(), count_pinned(turn));
    feats.insert("pinned_them".to_string(), count_pinned(opp));

    // Threats
    let count_threats = |side: Color| {
        let mut count = 0.0;
        let them = side.other();
        for sq in board.by_color(them) {
            let victim = board.piece_at(sq).unwrap();
            if victim.role == Role::King { continue; }
            let attackers = board.attacks_to(sq, side, occupied);
            for a_sq in attackers {
                if let Some(attacker) = board.piece_at(a_sq) {
                    if piece_value(attacker.role) < piece_value(victim.role) {
                        count += 1.0;
                    }
                }
            }
        }
        count
    };
    feats.insert("threats_us".to_string(), count_threats(turn));
    feats.insert("threats_them".to_string(), count_threats(opp));
}
