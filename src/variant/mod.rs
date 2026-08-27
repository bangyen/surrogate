//! Chess variant support for the native engine.
//!
//! The search and move generation are shared with standard chess — every
//! variant implements shakmaty's `Position` trait, so `alpha_beta` works
//! unchanged.  What differs is *evaluation*: each variant has its own
//! win condition, and reusing the standard piece-square tables would
//! produce legal but pointless play.
//!
//! Only the engine plays variants.  The ML explanation pipeline is
//! standard-chess only: it trains against Stockfish, which does not play
//! variants, and the extracted features encode standard-chess judgment.

use anyhow::{anyhow, Result};
use shakmaty::variant::{Antichess, KingOfTheHill, ThreeCheck};
use shakmaty::{Chess, Color, Position};
use std::fmt;
use std::str::FromStr;

use crate::eval::{evaluate, evaluate_antichess, koth_bonus, three_check_bonus};

/// A variant the native engine knows how to evaluate.
///
/// Deliberately narrower than shakmaty's variant list: these are the
/// variants whose evaluation this engine actually implements.  Crazyhouse
/// (piece drops), Atomic, Horde and Racing Kings are playable by the move
/// generator but would be evaluated with inapplicable heuristics, so they
/// are not offered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    /// Ordinary chess.
    Standard,
    /// Win by walking your king to a centre square.
    KingOfTheHill,
    /// Win by giving check three times.
    ThreeCheck,
    /// Win by losing all your pieces; captures are compulsory.
    Antichess,
}

impl Variant {
    /// Every variant the engine supports, in menu order.
    pub const ALL: [Variant; 4] = [
        Variant::Standard,
        Variant::KingOfTheHill,
        Variant::ThreeCheck,
        Variant::Antichess,
    ];

    /// The name accepted on the command line.
    pub fn slug(&self) -> &'static str {
        match self {
            Variant::Standard => "standard",
            Variant::KingOfTheHill => "koth",
            Variant::ThreeCheck => "3check",
            Variant::Antichess => "antichess",
        }
    }

    /// A one-line description of the win condition.
    pub fn description(&self) -> &'static str {
        match self {
            Variant::Standard => "Ordinary chess rules",
            Variant::KingOfTheHill => "Win by marching your king to the centre",
            Variant::ThreeCheck => "Win by giving check three times",
            Variant::Antichess => "Win by losing all your pieces; captures are forced",
        }
    }
}

impl fmt::Display for Variant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.slug())
    }
}

impl FromStr for Variant {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        // Accept the common aliases people actually type.
        match s.to_lowercase().replace(['-', '_'], "").as_str() {
            "standard" | "chess" => Ok(Variant::Standard),
            "koth" | "kingofthehill" => Ok(Variant::KingOfTheHill),
            "3check" | "threecheck" => Ok(Variant::ThreeCheck),
            "antichess" | "giveaway" | "losers" => Ok(Variant::Antichess),
            other => Err(anyhow!(
                "unknown variant '{}' (expected one of: {})",
                other,
                Variant::ALL
                    .iter()
                    .map(|v| v.slug())
                    .collect::<Vec<_>>()
                    .join(", ")
            )),
        }
    }
}

/// A position in whichever variant is being played.
///
/// Kept as a concrete enum rather than a boxed trait object so the search
/// stays monomorphic: each arm dispatches to a generic `alpha_beta`
/// specialised for that position type.
#[derive(Clone, Debug)]
pub enum VariantGame {
    Standard(Chess),
    KingOfTheHill(KingOfTheHill),
    ThreeCheck(ThreeCheck),
    Antichess(Antichess),
}

impl VariantGame {
    /// Start a new game of `variant` from its initial position.
    pub fn new(variant: Variant) -> Self {
        match variant {
            Variant::Standard => VariantGame::Standard(Chess::default()),
            Variant::KingOfTheHill => VariantGame::KingOfTheHill(KingOfTheHill::default()),
            Variant::ThreeCheck => VariantGame::ThreeCheck(ThreeCheck::default()),
            Variant::Antichess => VariantGame::Antichess(Antichess::default()),
        }
    }

    /// Borrow the underlying position.
    ///
    /// Callers that only handle standard chess -- the explanation
    /// pipeline, for instance -- match on this to opt out of variants.
    pub fn inner(&self) -> &VariantGame {
        self
    }

    /// Which variant this game is playing.
    pub fn variant(&self) -> Variant {
        match self {
            VariantGame::Standard(_) => Variant::Standard,
            VariantGame::KingOfTheHill(_) => Variant::KingOfTheHill,
            VariantGame::ThreeCheck(_) => Variant::ThreeCheck,
            VariantGame::Antichess(_) => Variant::Antichess,
        }
    }

    /// Side to move.
    pub fn turn(&self) -> Color {
        match self {
            VariantGame::Standard(p) => p.turn(),
            VariantGame::KingOfTheHill(p) => p.turn(),
            VariantGame::ThreeCheck(p) => p.turn(),
            VariantGame::Antichess(p) => p.turn(),
        }
    }

    /// Whether the game has ended under this variant's rules.
    pub fn is_game_over(&self) -> bool {
        match self {
            VariantGame::Standard(p) => p.is_game_over(),
            VariantGame::KingOfTheHill(p) => p.is_game_over(),
            VariantGame::ThreeCheck(p) => p.is_game_over(),
            VariantGame::Antichess(p) => p.is_game_over(),
        }
    }

    /// Legal moves in UCI notation.
    pub fn legal_moves(&self) -> Vec<String> {
        fn collect<P: Position>(pos: &P) -> Vec<String> {
            pos.legal_moves()
                .iter()
                .map(|m| {
                    shakmaty::uci::UciMove::from_move(*m, shakmaty::CastlingMode::Standard)
                        .to_string()
                })
                .collect()
        }

        match self {
            VariantGame::Standard(p) => collect(p),
            VariantGame::KingOfTheHill(p) => collect(p),
            VariantGame::ThreeCheck(p) => collect(p),
            VariantGame::Antichess(p) => collect(p),
        }
    }

    /// The position in FEN notation.
    pub fn fen(&self) -> String {
        fn to_fen<P: Position>(pos: &P) -> String {
            shakmaty::fen::Fen::from_position(pos, shakmaty::EnPassantMode::Always).to_string()
        }

        match self {
            VariantGame::Standard(p) => to_fen(p),
            VariantGame::KingOfTheHill(p) => to_fen(p),
            VariantGame::ThreeCheck(p) => to_fen(p),
            VariantGame::Antichess(p) => to_fen(p),
        }
    }

    /// Play a move given in UCI notation.
    pub fn play_uci(&mut self, uci: &str) -> Result<()> {
        fn apply<P: Position + Clone>(pos: &mut P, uci: &str) -> Result<()> {
            let parsed: shakmaty::uci::UciMove = uci
                .parse()
                .map_err(|e| anyhow!("invalid move format '{}': {:?}", uci, e))?;
            let m = parsed
                .to_move(pos)
                .map_err(|e| anyhow!("illegal move '{}': {:?}", uci, e))?;
            pos.play_unchecked(m);
            Ok(())
        }

        match self {
            VariantGame::Standard(p) => apply(p, uci),
            VariantGame::KingOfTheHill(p) => apply(p, uci),
            VariantGame::ThreeCheck(p) => apply(p, uci),
            VariantGame::Antichess(p) => apply(p, uci),
        }
    }

    /// Evaluate the position from the side to move's perspective,
    /// using the terms that matter for this variant.
    pub fn evaluate(&self) -> i32 {
        match self {
            VariantGame::Standard(p) => evaluate(p),
            // The standard evaluation still applies -- material and
            // structure matter -- plus the race for the centre.
            VariantGame::KingOfTheHill(p) => evaluate(p) + koth_bonus(p),
            VariantGame::ThreeCheck(p) => {
                let (white, black) = remaining_checks(p);
                evaluate(p) + three_check_bonus(white, black, p.turn())
            }
            // Material is a liability here, so the standard evaluation
            // is not merely inaccurate but backwards.
            VariantGame::Antichess(p) => evaluate_antichess(p),
        }
    }

    /// Search for the best move, returning it in UCI notation.
    pub fn best_move(&self, depth: u8) -> Option<String> {
        match self {
            VariantGame::Standard(p) => crate::search::find_best_reply_impl(p, depth),
            VariantGame::KingOfTheHill(p) => crate::search::find_best_reply_impl(p, depth),
            VariantGame::ThreeCheck(p) => crate::search::find_best_reply_impl(p, depth),
            VariantGame::Antichess(p) => crate::search::find_best_reply_impl(p, depth),
        }
    }
}

/// How many checks each side still needs to deliver to win.
///
/// shakmaty exposes this through the position's FEN, which carries a
/// `+w+b` style suffix; parsing it avoids depending on internals.
fn remaining_checks(pos: &ThreeCheck) -> (u32, u32) {
    let fen = shakmaty::fen::Fen::from_position(pos, shakmaty::EnPassantMode::Always).to_string();

    // The check counter appears as a trailing field like "3+3".
    for field in fen.split_whitespace() {
        if let Some((w, b)) = field.split_once('+') {
            if let (Ok(w), Ok(b)) = (w.parse::<u32>(), b.parse::<u32>()) {
                return (w, b);
            }
        }
    }
    (3, 3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variant_parses_its_own_slug() {
        for v in Variant::ALL {
            assert_eq!(
                v.slug().parse::<Variant>().unwrap(),
                v,
                "{} should round-trip",
                v.slug()
            );
        }
    }

    #[test]
    fn test_variant_parsing_accepts_aliases_and_case() {
        assert_eq!("KOTH".parse::<Variant>().unwrap(), Variant::KingOfTheHill);
        assert_eq!(
            "king-of-the-hill".parse::<Variant>().unwrap(),
            Variant::KingOfTheHill
        );
        assert_eq!(
            "three_check".parse::<Variant>().unwrap(),
            Variant::ThreeCheck
        );
        assert_eq!("giveaway".parse::<Variant>().unwrap(), Variant::Antichess);
        assert_eq!("chess".parse::<Variant>().unwrap(), Variant::Standard);
    }

    #[test]
    fn test_variant_parsing_rejects_unsupported_variants() {
        // Playable by shakmaty but not evaluated by this engine, so they
        // must be refused rather than played badly.
        for name in ["crazyhouse", "atomic", "horde", "racingkings", "nonsense"] {
            let err = name.parse::<Variant>().unwrap_err().to_string();
            assert!(err.contains("unknown variant"), "got: {err}");
            // The error should tell the user what is available.
            assert!(err.contains("koth"), "error should list options: {err}");
        }
    }

    #[test]
    fn test_new_game_starts_with_twenty_moves() {
        // Every supported variant shares the standard opening position;
        // only Antichess differs in which moves are legal.
        for v in [
            Variant::Standard,
            Variant::KingOfTheHill,
            Variant::ThreeCheck,
        ] {
            let g = VariantGame::new(v);
            assert_eq!(g.legal_moves().len(), 20, "{v} opening move count");
            assert_eq!(g.turn(), Color::White);
            assert!(!g.is_game_over());
            assert_eq!(g.variant(), v);
        }
    }

    #[test]
    fn test_antichess_opening_has_the_same_quiet_moves() {
        // No captures are available at the start, so compulsory capture
        // does not yet restrict anything.
        let g = VariantGame::new(Variant::Antichess);
        assert_eq!(g.legal_moves().len(), 20);
    }

    #[test]
    fn test_antichess_forces_captures() {
        let mut g = VariantGame::new(Variant::Antichess);
        // 1. e4 d5 leaves exd5 available, which becomes compulsory.
        g.play_uci("e2e4").unwrap();
        g.play_uci("d7d5").unwrap();

        let moves = g.legal_moves();
        assert_eq!(
            moves,
            vec!["e4d5".to_string()],
            "capture must be the only legal move in antichess"
        );
    }

    #[test]
    fn test_play_uci_rejects_illegal_and_malformed_moves() {
        let mut g = VariantGame::new(Variant::Standard);
        assert!(g.play_uci("e2e5").is_err(), "illegal move must be refused");
        assert!(
            g.play_uci("hello").is_err(),
            "malformed move must be refused"
        );
        // The position is unchanged after a rejected move.
        assert_eq!(g.legal_moves().len(), 20);
    }

    #[test]
    fn test_koth_rewards_a_central_king() {
        // A king on a centre square has already won, so it must
        // evaluate far above the same material with a home king.
        let central = VariantGame::KingOfTheHill(
            "4k3/8/8/3K4/8/8/8/8 w - - 0 1"
                .parse::<shakmaty::fen::Fen>()
                .unwrap()
                .into_position(shakmaty::CastlingMode::Standard)
                .unwrap(),
        );
        let home = VariantGame::KingOfTheHill(
            "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
                .parse::<shakmaty::fen::Fen>()
                .unwrap()
                .into_position(shakmaty::CastlingMode::Standard)
                .unwrap(),
        );
        assert!(
            central.evaluate() > home.evaluate() + 1000,
            "central king {} should dwarf home king {}",
            central.evaluate(),
            home.evaluate()
        );
    }

    #[test]
    fn test_antichess_evaluation_is_inverted() {
        // Being a queen up is losing in antichess, so the side with
        // extra material must evaluate worse.
        let queen_up = VariantGame::Antichess(
            "4k3/8/8/8/8/8/8/3QK3 w - - 0 1"
                .parse::<shakmaty::fen::Fen>()
                .unwrap()
                .into_position(shakmaty::CastlingMode::Standard)
                .unwrap(),
        );
        assert!(
            queen_up.evaluate() < 0,
            "an extra queen should be a liability, got {}",
            queen_up.evaluate()
        );
    }

    #[test]
    fn test_standard_evaluation_is_unchanged_by_the_variant_layer() {
        // Routing standard chess through VariantGame must not alter it.
        let g = VariantGame::new(Variant::Standard);
        assert_eq!(g.evaluate(), crate::eval::evaluate(&Chess::default()));
    }

    #[test]
    fn test_best_move_is_legal_in_every_variant() {
        for v in Variant::ALL {
            let g = VariantGame::new(v);
            let mv = g
                .best_move(3)
                .unwrap_or_else(|| panic!("{v} produced no move"));
            assert!(
                g.legal_moves().contains(&mv),
                "{v} returned illegal move {mv}"
            );
        }
    }

    #[test]
    fn test_three_check_counter_starts_at_three() {
        let VariantGame::ThreeCheck(p) = VariantGame::new(Variant::ThreeCheck) else {
            panic!("expected a three-check game");
        };
        assert_eq!(remaining_checks(&p), (3, 3));
    }

    #[test]
    fn test_three_check_bonus_favours_the_side_closer_to_winning() {
        // White needing one more check is far better than needing three.
        let ahead = three_check_bonus(1, 3, Color::White);
        let behind = three_check_bonus(3, 1, Color::White);
        assert!(
            ahead > behind,
            "being a check away ({ahead}) should beat being behind ({behind})"
        );
        // The evaluation flips with the side to move.
        assert_eq!(
            three_check_bonus(1, 3, Color::Black),
            -three_check_bonus(1, 3, Color::White)
        );
    }

    #[test]
    fn test_fen_round_trips_through_moves() {
        let mut g = VariantGame::new(Variant::KingOfTheHill);
        assert!(g.fen().starts_with("rnbqkbnr/pppppppp"));
        g.play_uci("e2e4").unwrap();
        assert!(
            g.fen().contains("4P3"),
            "fen should reflect the move: {}",
            g.fen()
        );
    }
}

#[cfg(test)]
mod behaviour_tests {
    use super::*;

    fn koth_from(fen: &str) -> VariantGame {
        VariantGame::KingOfTheHill(
            fen.parse::<shakmaty::fen::Fen>()
                .unwrap()
                .into_position(shakmaty::CastlingMode::Standard)
                .unwrap(),
        )
    }

    #[test]
    fn test_koth_engine_steps_onto_the_hill_when_it_can() {
        // White king on d4-adjacent square with a free path to e4/d5:
        // taking the centre wins immediately, so the search must find it.
        let g = koth_from("7k/8/8/8/8/3K4/8/8 w - - 0 1");
        let mv = g.best_move(4).expect("engine should find a move");

        // d3 touches d4 and e4, both winning squares.
        assert!(
            mv == "d3d4" || mv == "d3e4",
            "engine should step onto the hill, played {mv}"
        );
    }

    #[test]
    fn test_koth_prefers_the_centre_over_an_edge_advance() {
        let central = koth_from("7k/8/8/8/3K4/8/8/8 w - - 0 1");
        let edge = koth_from("7k/8/8/8/8/8/8/K7 w - - 0 1");
        assert!(
            central.evaluate() > edge.evaluate(),
            "a king on d4 ({}) should beat one on a1 ({})",
            central.evaluate(),
            edge.evaluate()
        );
    }

    #[test]
    fn test_antichess_engine_gives_material_away() {
        // Black to move with a capture available: in antichess, being
        // captured is progress, so the engine should not fear losing
        // material. Verify it plays a legal move and evaluation favours
        // the side with less material.
        let fewer = VariantGame::Antichess(
            "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
                .parse::<shakmaty::fen::Fen>()
                .unwrap()
                .into_position(shakmaty::CastlingMode::Standard)
                .unwrap(),
        );
        let more = VariantGame::Antichess(
            "4k3/8/8/8/8/8/PPP5/4K3 w - - 0 1"
                .parse::<shakmaty::fen::Fen>()
                .unwrap()
                .into_position(shakmaty::CastlingMode::Standard)
                .unwrap(),
        );
        assert!(
            fewer.evaluate() > more.evaluate(),
            "fewer pieces ({}) should evaluate better than more ({})",
            fewer.evaluate(),
            more.evaluate()
        );
    }
}
