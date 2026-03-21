use shakmaty::{Chess, Color, Position, Role, Square};
use std::collections::BTreeMap;

pub fn extract(pos: &Chess, feats: &mut BTreeMap<String, f32>, turn: Color, _opp: Color) {
    let board = pos.board();
    let mut mat_us = 0.0;
    let mut mat_them = 0.0;
    let mut phase = 0;

    for sq in Square::ALL {
        if let Some(piece) = board.piece_at(sq) {
            let val = match piece.role {
                Role::Pawn => 1.0,
                Role::Knight => 3.0,
                Role::Bishop => 3.1,
                Role::Rook => 5.0,
                Role::Queen => 9.0,
                Role::King => 0.0,
            };
            if piece.color == turn {
                mat_us += val;
            } else {
                mat_them += val;
            }

            if piece.role != Role::Pawn && piece.role != Role::King {
                phase += 1;
            }
        }
    }
    feats.insert("material_us".to_string(), mat_us);
    feats.insert("material_them".to_string(), mat_them);
    feats.insert("material_diff".to_string(), mat_us - mat_them);
    feats.insert("phase".to_string(), phase as f32);
}
