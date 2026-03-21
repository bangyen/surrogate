pub mod king_safety;
pub mod material;
pub mod mobility;
pub mod pawn_structure;
pub mod positional;
pub mod tactical;

use shakmaty::{Chess, Position};
use std::collections::BTreeMap;

pub fn extract_features(pos: &Chess) -> BTreeMap<String, f32> {
    let mut feats = BTreeMap::new();
    let turn = pos.turn();
    let opp = turn.other();

    material::extract(pos, &mut feats, turn, opp);
    mobility::extract(pos, &mut feats, turn, opp);
    king_safety::extract(pos, &mut feats, turn, opp);
    pawn_structure::extract(pos, &mut feats, turn, opp);
    tactical::extract(pos, &mut feats, turn, opp);
    positional::extract(pos, &mut feats, turn, opp);

    feats
}
