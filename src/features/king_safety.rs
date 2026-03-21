use shakmaty::{Bitboard, Chess, Color, Position, Role, Square};
use std::collections::BTreeMap;

pub fn extract(pos: &Chess, feats: &mut BTreeMap<String, f32>, turn: Color, opp: Color) {
    let board = pos.board();
    let phase = feats.get("phase").cloned().unwrap_or(24.0);

    // King Ring Pressure
    let get_king_ring = |side: Color| {
        let mut ring = Bitboard::EMPTY;
        if let Some(ksq) = board.king_of(side) {
            for s in Square::ALL {
                if ksq.distance(s) <= 1 {
                    ring |= Bitboard::from_square(s);
                }
            }
        }
        ring
    };

    let weight_pressure = |role: Role| match role {
        Role::Pawn => 1.0,
        Role::Knight => 3.0f32.powf(0.7),
        Role::Bishop => 3.1f32.powf(0.7),
        Role::Rook => 5.0f32.powf(0.7),
        Role::Queen => 9.0f32.powf(0.7),
        _ => 0.0,
    };

    let calc_pressure = |attacking_side: Color| {
        let ring = get_king_ring(attacking_side.other());
        if ring.is_empty() {
            return 0.0;
        }
        let mut s = 0.0;
        let occupied = board.occupied();
        for sq in ring {
            let attackers = board.attacks_to(sq, attacking_side, occupied);
            if !attackers.is_empty() {
                let mut max_w = 0.0;
                for a_sq in attackers {
                    if let Some(p) = board.piece_at(a_sq) {
                        let w = weight_pressure(p.role);
                        if w > max_w {
                            max_w = w;
                        }
                    }
                }
                s += max_w;
            }
        }
        s / (phase).max(6.0)
    };

    feats.insert("king_ring_pressure_us".to_string(), calc_pressure(turn));
    feats.insert("king_ring_pressure_them".to_string(), calc_pressure(opp));

    // King Safety
    let king_safety = |side: Color| {
        if let Some(ksq) = board.king_of(side) {
            let mut safety = 0.0;
            for sq in board.attacks_from(ksq) {
                if let Some(p) = board.piece_at(sq) {
                    if p.color == side {
                        safety += 1.0;
                    }
                }
            }
            safety
        } else {
            0.0
        }
    };
    feats.insert("king_safety_us".to_string(), king_safety(turn));
    feats.insert("king_safety_them".to_string(), king_safety(opp));

    // King Pawn Shield
    let king_pawn_shield = |side: Color| {
        if let Some(ksq) = board.king_of(side) {
            let file = ksq.file();
            let rank = ksq.rank();
            let mut count = 0;

            let shield_ranks = match side {
                Color::White => [rank.offset(1), rank.offset(2)],
                Color::Black => [rank.offset(-1), rank.offset(-2)],
            };

            let mut shield_files = Vec::new();
            if let Some(f) = file.offset(-1) {
                shield_files.push(f);
            }
            shield_files.push(file);
            if let Some(f) = file.offset(1) {
                shield_files.push(f);
            }

            for &f in &shield_files {
                for &r_opt in &shield_ranks {
                    if let Some(r) = r_opt {
                        let sq = Square::from_coords(f, r);
                        if let Some(p) = board.piece_at(sq) {
                            if p.role == Role::Pawn && p.color == side {
                                count += 1;
                            }
                        }
                    }
                }
            }
            count as f32
        } else {
            0.0
        }
    };
    feats.insert("king_pawn_shield_us".to_string(), king_pawn_shield(turn));
    feats.insert("king_pawn_shield_them".to_string(), king_pawn_shield(opp));
}
