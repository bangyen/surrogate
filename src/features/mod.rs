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

#[cfg(test)]
mod tests {
    use super::*;
    use shakmaty::fen::Fen;
    use shakmaty::CastlingMode;

    fn pos_from_fen(fen: &str) -> Chess {
        let setup: Fen = fen.parse().unwrap();
        setup.into_position(CastlingMode::Standard).unwrap()
    }

    const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    fn get(feats: &BTreeMap<String, f32>, key: &str) -> f32 {
        *feats
            .get(key)
            .unwrap_or_else(|| panic!("missing feature {key}"))
    }

    #[test]
    fn test_extract_features_is_symmetric_at_start() {
        // The initial position is mirror-symmetric, so every "us"/"them"
        // pair must agree and all diffs must vanish.
        let feats = extract_features(&pos_from_fen(START));

        for (name, val) in &feats {
            if let Some(stem) = name.strip_suffix("_us") {
                let them = format!("{stem}_them");
                if let Some(other) = feats.get(&them) {
                    // Tolerance covers f32 summation order, not asymmetry.
                    assert!(
                        (val - other).abs() < 1e-4,
                        "{name} ({val}) != {them} ({other}) in the symmetric start position"
                    );
                }
            }
            if name.ends_with("_diff") {
                assert!(
                    val.abs() < 1e-4,
                    "{name} should be 0 at the start, got {val}"
                );
            }
        }
    }

    #[test]
    fn test_extract_features_is_deterministic() {
        let pos = pos_from_fen(START);
        assert_eq!(extract_features(&pos), extract_features(&pos));
    }

    #[test]
    fn test_material_counts_and_phase() {
        let feats = extract_features(&pos_from_fen(START));
        // 8 pawns + 2N(3) + 2B(3.1) + 2R(5) + Q(9) = 8 + 6 + 6.2 + 10 + 9
        assert_eq!(get(&feats, "material_us"), 39.2);
        assert_eq!(get(&feats, "material_them"), 39.2);
        assert_eq!(get(&feats, "material_diff"), 0.0);
        // Phase counts non-pawn, non-king pieces: 2*(2+2+2+1) = 14.
        assert_eq!(get(&feats, "phase"), 14.0);
    }

    #[test]
    fn test_material_is_relative_to_side_to_move() {
        // Black is a queen up, and it is Black to move.
        let feats = extract_features(&pos_from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR b KQkq - 0 1",
        ));
        assert_eq!(get(&feats, "material_diff"), 9.0);

        // Same board, White to move: the diff flips sign.
        let feats = extract_features(&pos_from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1",
        ));
        assert_eq!(get(&feats, "material_diff"), -9.0);
    }

    #[test]
    fn test_phase_drops_in_the_endgame() {
        let start = extract_features(&pos_from_fen(START));
        let endgame = extract_features(&pos_from_fen("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"));
        assert_eq!(get(&endgame, "phase"), 0.0);
        assert!(get(&start, "phase") > get(&endgame, "phase"));
    }

    #[test]
    fn test_mobility_reflects_open_position() {
        // A lone queen in the open has far more moves than the cramped
        // start position's side to move.
        let start = extract_features(&pos_from_fen(START));
        let open = extract_features(&pos_from_fen("4k3/8/8/3Q4/8/8/8/4K3 w - - 0 1"));
        assert!(
            get(&open, "mobility_us") > get(&start, "mobility_us"),
            "open queen mobility {} should exceed start mobility {}",
            get(&open, "mobility_us"),
            get(&start, "mobility_us")
        );
    }

    #[test]
    fn test_pawn_structure_detects_doubled_and_isolated() {
        let feats = extract_features(&pos_from_fen(START));
        // The initial pawn wall has neither doubled nor isolated pawns.
        assert_eq!(get(&feats, "doubled_pawns_us"), 0.0);
        assert_eq!(get(&feats, "isolated_pawns_us"), 0.0);

        // Two white pawns stacked on the a-file with no b-file neighbour:
        // doubled and isolated.
        let feats = extract_features(&pos_from_fen("4k3/8/8/8/P7/P7/8/4K3 w - - 0 1"));
        assert!(
            get(&feats, "doubled_pawns_us") > 0.0,
            "stacked a-pawns should register as doubled"
        );
        assert!(
            get(&feats, "isolated_pawns_us") > 0.0,
            "a-pawns with no b-pawn should register as isolated"
        );
    }

    #[test]
    fn test_all_features_are_finite() {
        // NaN or infinity here would silently poison model training.
        for fen in [
            START,
            "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
            "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1",
            "8/2k5/8/8/8/8/5K2/8 w - - 0 1",
        ] {
            for (name, val) in extract_features(&pos_from_fen(fen)) {
                assert!(val.is_finite(), "feature {name} was {val} for FEN {fen}");
            }
        }
    }
}
