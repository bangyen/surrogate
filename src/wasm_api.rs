//! Browser bindings.
//!
//! Exposes the native engine, feature extraction and surrogate
//! explanations to JavaScript.  Everything here runs in the page: there
//! is no Stockfish subprocess and no server round trip, so the opponent
//! is this project's own alpha-beta search rather than Stockfish.
//!
//! The surrogate model is trained offline against Stockfish and shipped
//! as a JSON asset, so explanations reflect what that model learned.

use wasm_bindgen::prelude::*;

use crate::features::extract_features;
use crate::ml::{PhaseEnsemble, SurrogateExplainer};
use crate::variant::{Variant, VariantGame};

/// A game the browser can drive.
#[wasm_bindgen]
pub struct WasmGame {
    game: VariantGame,
    explainer: Option<SurrogateExplainer>,
}

#[wasm_bindgen]
impl WasmGame {
    /// Start a new game.  `variant` accepts the same names as the CLI
    /// ("standard", "koth", "3check", "antichess").
    #[wasm_bindgen(constructor)]
    pub fn new(variant: &str) -> Result<WasmGame, JsError> {
        let variant: Variant = variant.parse().map_err(|e| JsError::new(&format!("{e}")))?;
        Ok(WasmGame {
            game: VariantGame::new(variant),
            explainer: None,
        })
    }

    /// Load the surrogate model, enabling explanations.
    ///
    /// Without it the engine still plays; moves simply come back without
    /// reasons attached.
    #[wasm_bindgen(js_name = loadModel)]
    pub fn load_model(&mut self, model_json: &str) -> Result<(), JsError> {
        let model: PhaseEnsemble = serde_json::from_str(model_json)
            .map_err(|e| JsError::new(&format!("invalid model: {e}")))?;
        self.explainer = Some(SurrogateExplainer::new(model));
        Ok(())
    }

    /// Whether explanations are available.
    #[wasm_bindgen(js_name = hasModel)]
    pub fn has_model(&self) -> bool {
        self.explainer.is_some()
    }

    /// The current position in FEN notation.
    pub fn fen(&self) -> String {
        self.game.fen()
    }

    /// Legal moves in UCI notation.
    #[wasm_bindgen(js_name = legalMoves)]
    pub fn legal_moves(&self) -> Vec<String> {
        self.game.legal_moves()
    }

    /// Whether the game has ended under this variant's rules.
    #[wasm_bindgen(js_name = isGameOver)]
    pub fn is_game_over(&self) -> bool {
        self.game.is_game_over()
    }

    /// "white" or "black".
    pub fn turn(&self) -> String {
        if self.game.turn().is_white() {
            "white".to_string()
        } else {
            "black".to_string()
        }
    }

    /// Static evaluation from the side to move's perspective, in
    /// centipawns.
    pub fn evaluate(&self) -> i32 {
        self.game.evaluate()
    }

    /// Play a move given in UCI notation.
    #[wasm_bindgen(js_name = playMove)]
    pub fn play_move(&mut self, uci: &str) -> Result<(), JsError> {
        self.game
            .play_uci(uci)
            .map_err(|e| JsError::new(&format!("{e}")))
    }

    /// Search for the engine's best move *without* playing it, returning
    /// UCI notation.  Returns `None` when there is no legal move.
    ///
    /// Searching and playing are separate so a caller can explain the
    /// move first: `explain` describes a move from the position it is
    /// played in, which is gone once the move has been applied.
    ///
    /// This runs on the caller's thread, so keep `depth` modest -- the
    /// browser is unresponsive until it returns.
    #[wasm_bindgen(js_name = searchMove)]
    pub fn search_move(&self, depth: u8) -> Option<String> {
        self.game.best_move(depth)
    }

    /// Search for the engine's reply and play it in one step.
    #[wasm_bindgen(js_name = playEngineMove)]
    pub fn play_engine_move(&mut self, depth: u8) -> Result<Option<String>, JsError> {
        let Some(mv) = self.game.best_move(depth) else {
            return Ok(None);
        };
        self.game
            .play_uci(&mv)
            .map_err(|e| JsError::new(&format!("engine produced an illegal move: {e}")))?;
        Ok(Some(mv))
    }

    /// Explain a move without playing it, as newline-separated reasons.
    ///
    /// Returns an empty string when no model is loaded or the surrogate
    /// has nothing above the noise floor to say.
    pub fn explain(&self, uci: &str) -> Result<String, JsError> {
        let Some(explainer) = &self.explainer else {
            return Ok(String::new());
        };

        let mut after = self.game.clone();
        after
            .play_uci(uci)
            .map_err(|e| JsError::new(&format!("{e}")))?;

        // Explanations are defined for standard chess: the surrogate was
        // trained on Stockfish evaluations of ordinary positions, and the
        // features encode standard-chess judgment.
        let VariantGame::Standard(pos) = &after.inner() else {
            return Ok(String::new());
        };

        let feats = extract_features(pos);
        let reasons = explainer.explain_move(&feats, 3, 0.05);
        Ok(reasons
            .iter()
            .map(|(_, _, text)| text.clone())
            .collect::<Vec<_>>()
            .join("\n"))
    }

    /// Extracted features for the current position, as JSON.
    #[wasm_bindgen(js_name = featuresJson)]
    pub fn features_json(&self) -> Result<String, JsError> {
        let VariantGame::Standard(pos) = &self.game.inner() else {
            return Ok("{}".to_string());
        };
        serde_json::to_string(&extract_features(pos)).map_err(|e| JsError::new(&format!("{e}")))
    }
}

/// Variants the browser build can offer, as `slug\tdescription` lines.
#[wasm_bindgen(js_name = supportedVariants)]
pub fn supported_variants() -> Vec<String> {
    Variant::ALL
        .iter()
        .map(|v| format!("{}\t{}", v.slug(), v.description()))
        .collect()
}
