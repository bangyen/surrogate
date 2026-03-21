use shakmaty::{Bitboard, Chess, Color, Position, Role};
use std::collections::BTreeMap;

pub fn extract(pos: &Chess, feats: &mut BTreeMap<String, f32>, _turn: Color, _opp: Color) {
    // Basic Mobility
    let moves = pos.legal_moves();
    feats.insert("mobility_us".to_string(), (moves.len() as f32).min(40.0));

    let opp_pos = pos.clone().swap_turn().unwrap_or_else(|_| pos.clone());
    feats.insert(
        "mobility_them".to_string(),
        (opp_pos.legal_moves().len() as f32).min(40.0),
    );

    // Safe Mobility
    feats.insert("safe_mobility_us".to_string(), get_safe_mobility(pos));
    feats.insert(
        "safe_mobility_them".to_string(),
        get_safe_mobility(&opp_pos),
    );
}

fn get_safe_mobility(pos_in: &Chess) -> f32 {
    let side = pos_in.turn();
    let opp = side.other();
    let mut enemy_pawn_attacks = Bitboard::EMPTY;
    for sq in pos_in.board().by_role(Role::Pawn) & pos_in.board().by_color(opp) {
        enemy_pawn_attacks |= shakmaty::attacks::pawn_attacks(opp, sq);
    }

    let mut safe_count = 0;
    for m in pos_in.legal_moves() {
        if !(enemy_pawn_attacks & Bitboard::from_square(m.to())).is_empty() {
            continue;
        }
        safe_count += 1;
    }
    (safe_count as f32).min(40.0)
}
