use anyhow::{anyhow, Result};
use linfa::prelude::*;
use linfa_elasticnet::ElasticNet;
use ndarray::{Array1, Array2, Axis};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use shakmaty::{Chess, Position, Role, Square};

use crate::engine::UciEngine;
use crate::features::extract_features;
use crate::ml::model::{PhaseEnsemble, PhaseModel};
use crate::ml::scaler::StandardScaler;

/// Largest evaluation magnitude kept in training targets.
///
/// Stockfish reports forced mates near ±10000.  Random-play positions
/// contain plenty of them, and letting those through makes the surrogate
/// fit a handful of enormous targets at the expense of the ordinary
/// ±50 cp swings the explanations are actually about.
pub const EVAL_CLIP_CP: i32 = 1000;

/// Clamp an engine evaluation into the range the surrogate is fit over.
pub fn clip_eval(cp: i32) -> i32 {
    cp.clamp(-EVAL_CLIP_CP, EVAL_CLIP_CP)
}

/// Bound on a single training target, applied after the delta is formed.
///
/// Even with mate scores clipped, a handful of positions swing by many
/// hundreds of centipawns and dominate a squared-error fit.  Bounding the
/// target keeps the model fitting the ordinary positions the
/// explanations are about.  Applied to the audit's targets too, so both
/// measure the same quantity.
pub const TARGET_CLIP_CP: f64 = 400.0;

/// How much predictive accuracy to trade for a readable explanation.
///
/// Cross-validation scores squared error only, so it will choose a dense
/// ridge fit that spreads an explanation across dozens of correlated
/// features.  Configurations within this fraction of the best error are
/// treated as equivalent, and the sparsest among them wins.
///
/// Calibrated by measurement: a 2% tolerance drove the model down to
/// four coefficients and pushed move-ranking correlation below zero,
/// which is worse than useless.  Half a percent keeps the preference
/// without letting it override accuracy.
pub const SPARSITY_TOLERANCE: f64 = 0.005;

/// Rows required per feature before a phase gets its own model.
///
/// Below this the phase fit is noisier than the global model it would
/// replace, so the ensemble falls back to the global coefficients.
pub const MIN_ROWS_PER_PHASE_FEATURE: usize = 8;

/// Clamp a training target into the range the surrogate is fit over.
pub fn clip_target(delta: f64) -> f64 {
    delta.clamp(-TARGET_CLIP_CP, TARGET_CLIP_CP)
}

pub fn board_phase_value(pos: &Chess) -> i32 {
    let board = pos.board();
    let mut val = 0;
    for sq in Square::ALL {
        if let Some(piece) = board.piece_at(sq) {
            match piece.role {
                Role::Queen | Role::Rook | Role::Bishop | Role::Knight => val += 1,
                _ => {}
            }
        }
    }
    val
}

pub fn classify_phase(pos: &Chess) -> String {
    let phase = board_phase_value(pos);
    if phase >= 12 {
        "opening".to_string()
    } else if phase >= 6 {
        "middlegame".to_string()
    } else {
        "endgame".to_string()
    }
}

pub fn generate_stratified_positions(n: usize) -> Vec<Chess> {
    generate_stratified_positions_seeded(n, rand::thread_rng().gen())
}

/// Sample positions from a given seed.
///
/// The audit needs this: sampling fresh positions every run makes two
/// measurements incomparable, and with only a few dozen decisive
/// comparisons per run the sampling noise swamps real differences
/// between models.
pub fn generate_stratified_positions_seeded(n: usize, seed: u64) -> Vec<Chess> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let targets = [
        ("opening", (n as f32 * 0.25) as usize),
        ("middlegame", (n as f32 * 0.50) as usize),
        ("endgame", (n as f32 * 0.25) as usize),
    ];

    let mut boards = Vec::new();
    for (phase, target) in targets {
        let mut count = 0;
        let mut attempts = 0;
        let max_attempts = target * 50;

        while count < target && attempts < max_attempts {
            attempts += 1;
            let mut b = Chess::default();
            let plies = match phase {
                "opening" => rng.gen_range(4..14),
                "middlegame" => rng.gen_range(15..35),
                _ => rng.gen_range(36..60),
            };

            for _ in 0..plies {
                let moves = b.legal_moves();
                if moves.is_empty() || b.is_game_over() {
                    break;
                }

                if phase == "endgame" {
                    let captures: Vec<_> = moves.iter().filter(|m| m.is_capture()).collect();
                    if !captures.is_empty() && rng.gen_bool(0.6) {
                        b.play_unchecked(**captures.choose(&mut rng).unwrap());
                        continue;
                    }
                }
                b.play_unchecked(**moves.iter().collect::<Vec<_>>().choose(&mut rng).unwrap());
            }

            if !b.is_game_over() && classify_phase(&b) == phase {
                boards.push(b);
                count += 1;
            }
        }
    }
    boards.shuffle(&mut rng);
    boards
}

pub fn train_surrogate_model(engine_path: &str, n_positions: usize) -> Result<PhaseEnsemble> {
    let mut engine = UciEngine::new(engine_path)?;
    let boards = generate_stratified_positions(n_positions);

    let mut x_raw = Vec::new();
    let mut y_raw = Vec::new();
    let mut feature_names = Vec::new();
    let mut set_feature_names = false;
    let mut row_phase: Vec<String> = Vec::new();
    let mut dump_fens: Vec<String> = Vec::new();

    for (i, b) in boards.iter().enumerate() {
        if (i + 1) % 10 == 0 {
            println!("Processing position {}/{}...", i + 1, n_positions);
        }

        let fen = shakmaty::fen::Fen::from_position(b, shakmaty::EnPassantMode::Always).to_string();
        let base_eval_res = engine.get_evaluation(&fen, 12);
        if base_eval_res.is_err() {
            continue;
        }
        let base_eval = base_eval_res.unwrap();

        let top_moves_res = engine.get_top_moves(&fen, 12, 3);
        if top_moves_res.is_err() {
            continue;
        }
        let top_moves = top_moves_res.unwrap();

        for (mv_uci, _score) in top_moves {
            let mut b_after = b.clone();
            let uci_move: shakmaty::uci::UciMove = match mv_uci.parse() {
                Ok(m) => m,
                Err(_) => continue,
            };

            if let Ok(m) = uci_move.to_move(b) {
                b_after.play_unchecked(m);

                let fen_after =
                    shakmaty::fen::Fen::from_position(&b_after, shakmaty::EnPassantMode::Always)
                        .to_string();
                let best_reply_uci_res = engine.get_best_move(&fen_after, 12);
                if best_reply_uci_res.is_err() {
                    continue;
                }
                let best_reply_uci = best_reply_uci_res.unwrap();

                let reply_uci: shakmaty::uci::UciMove = match best_reply_uci.parse() {
                    Ok(m) => m,
                    Err(_) => continue,
                };

                if let Ok(rm) = reply_uci.to_move(&b_after) {
                    b_after.play_unchecked(rm);

                    let fen_final = shakmaty::fen::Fen::from_position(
                        &b_after,
                        shakmaty::EnPassantMode::Always,
                    )
                    .to_string();
                    let after_eval_res = engine.get_evaluation(&fen_final, 12);
                    if after_eval_res.is_err() {
                        continue;
                    }
                    let after_eval = after_eval_res.unwrap();

                    let delta = clip_eval(after_eval) - clip_eval(base_eval);

                    let feats = extract_features(&b_after);
                    if !set_feature_names {
                        feature_names = feats.keys().cloned().collect();
                        set_feature_names = true;
                    }

                    let mut row = Vec::new();
                    for name in &feature_names {
                        row.push(*feats.get(name).unwrap_or(&0.0) as f64);
                    }
                    x_raw.push(row);
                    y_raw.push(clip_target(delta as f64));
                    dump_fens.push(fen_final.clone());
                    // Each position contributes one row per candidate
                    // move, so remember which board each row came from.
                    row_phase.push(classify_phase(b));
                }
            }
        }
    }

    if x_raw.is_empty() {
        return Err(anyhow!("No training data collected"));
    }

    // Optional dump for offline analysis of the training set.
    if let Ok(path) = std::env::var("CHESS_AI_DUMP_TRAINING") {
        if let Ok(json) =
            serde_json::to_string(&(&x_raw, &y_raw, &feature_names, &row_phase, &dump_fens))
        {
            let _ = std::fs::write(&path, json);
            println!("Dumped training data to {}", path);
        }
    }

    let n_samples = x_raw.len();
    let n_features = feature_names.len();
    let x_mat = Array2::from_shape_vec(
        (n_samples, n_features),
        x_raw.into_iter().flatten().collect(),
    )?;
    let y_vec = Array1::from_vec(y_raw);

    let mut scaler = StandardScaler::new(n_features);
    scaler.fit(&x_mat);
    let x_scaled = scaler.transform(&x_mat);

    let mut ensemble = PhaseEnsemble::new(feature_names);
    ensemble.scaler = Some(scaler);

    println!("Training global model with {} samples...", n_samples);
    let dataset = Dataset::new(x_scaled.clone(), y_vec.clone());
    let (best_alpha, best_l1) = cross_validate_elastic_net(&dataset)?;

    let global_model = ElasticNet::params()
        .penalty(best_alpha)
        .l1_ratio(best_l1)
        .fit(&dataset)
        .map_err(|e| anyhow!("Failed to fit global model: {}", e))?;

    ensemble.global_model = Some(PhaseModel {
        coefficients: global_model.hyperplane().clone(),
        intercept: global_model.intercept(),
        alpha: best_alpha,
        l1_ratio: best_l1,
    });

    for phase_name in ["opening", "middlegame", "endgame"] {
        // Rows are per (position, move), so the phase of row `i` comes
        // from the recorded source board, not from `boards[i]`.
        let idx: Vec<usize> = (0..n_samples)
            .filter(|&i| row_phase[i] == phase_name)
            .collect();

        // A phase model must have enough rows to beat the global model
        // it replaces.  Measured on a 445-row sample, splitting three
        // ways scored R2 0.130 against the global model's 0.156: each
        // phase was too data-starved to justify its own fit.  Require
        // several rows per feature before specialising.
        if idx.len() >= MIN_ROWS_PER_PHASE_FEATURE * n_features {
            println!(
                "Training model for {} ({} samples)...",
                phase_name,
                idx.len()
            );
            let x_phase = x_scaled.select(Axis(0), &idx);
            let y_phase = y_vec.select(Axis(0), &idx);
            let ds_phase = Dataset::new(x_phase, y_phase);
            let (pa, pl1) = cross_validate_elastic_net(&ds_phase)?;
            let m = ElasticNet::params()
                .penalty(pa)
                .l1_ratio(pl1)
                .fit(&ds_phase)
                .map_err(|e| anyhow!("Failed to fit phase model: {}", e))?;

            ensemble.models.insert(
                phase_name.to_string(),
                PhaseModel {
                    coefficients: m.hyperplane().clone(),
                    intercept: m.intercept(),
                    alpha: pa,
                    l1_ratio: pl1,
                },
            );
        }
    }

    Ok(ensemble)
}

/// Choose ElasticNet hyper-parameters by k-fold cross-validation.
///
/// Rows arrive in blocks -- several candidate moves per sampled position,
/// grouped by phase -- so the dataset is shuffled before folding.
/// Contiguous folds would otherwise validate against positions closely
/// related to the ones just trained on.
fn cross_validate_elastic_net(dataset: &Dataset<f64, f64, ndarray::Ix1>) -> Result<(f64, f64)> {
    let alphas = [0.01, 0.1, 1.0, 10.0, 30.0, 100.0, 300.0, 1000.0];
    // Weighted toward L1: cross-validation optimises squared error alone
    // and will happily pick a dense ridge fit, but an explanation the
    // reader cannot follow has no value here.  See SPARSITY_TOLERANCE.
    let l1_ratios = [0.5, 0.7, 0.9, 1.0];

    let n = dataset.nsamples();
    if n < 10 {
        return Ok((0.1, 1.0)); // Default params for very small data
    }

    let mut rng = rand::rngs::StdRng::seed_from_u64(0xC0FFEE);
    let x = dataset.records();
    let y = dataset.targets();
    let mut order: Vec<usize> = (0..n).collect();
    order.shuffle(&mut rng);

    let k = 5.min(n);
    let mut candidates: Vec<(f64, f64, f64)> = Vec::new();

    for &a in &alphas {
        for &l1 in &l1_ratios {
            let mut total_err = 0.0;
            let mut scored = 0usize;

            for fold in 0..k {
                // Disjoint folds: every row validates exactly once.
                let val_idx: Vec<usize> = order
                    .iter()
                    .enumerate()
                    .filter(|(pos, _)| pos % k == fold)
                    .map(|(_, &i)| i)
                    .collect();
                let train_idx: Vec<usize> = order
                    .iter()
                    .enumerate()
                    .filter(|(pos, _)| pos % k != fold)
                    .map(|(_, &i)| i)
                    .collect();

                if val_idx.is_empty() || train_idx.is_empty() {
                    continue;
                }

                let train =
                    Dataset::new(x.select(Axis(0), &train_idx), y.select(Axis(0), &train_idx));

                let Ok(m) = ElasticNet::params().penalty(a).l1_ratio(l1).fit(&train) else {
                    continue;
                };

                let x_val = x.select(Axis(0), &val_idx);
                let preds = m.predict(&x_val);
                for (p, i) in preds.iter().zip(&val_idx) {
                    total_err += (p - y[*i]).powi(2);
                    scored += 1;
                }
            }

            // Normalise per validation row so configurations that scored
            // different numbers of rows stay comparable.
            if scored == 0 {
                continue;
            }
            let avg_err = total_err / scored as f64;
            candidates.push((avg_err, a, l1));
        }
    }
    if candidates.is_empty() {
        return Ok((0.1, 1.0));
    }

    // Among configurations that predict about as well as the best one,
    // prefer the sparsest.  The surrogate exists to be read, and a dense
    // fit spreads its explanation across dozens of correlated features
    // for a negligible gain in squared error.
    let best_err = candidates
        .iter()
        .map(|(e, _, _)| *e)
        .fold(f64::MAX, f64::min);
    let cutoff = best_err * (1.0 + SPARSITY_TOLERANCE);

    let (_, alpha, l1) = candidates
        .iter()
        .filter(|(e, _, _)| *e <= cutoff)
        // Higher l1_ratio zeroes more coefficients; break ties on the
        // stronger penalty, which also shrinks the model.
        .max_by(|x, y| {
            x.2.partial_cmp(&y.2)
                .unwrap()
                .then(x.1.partial_cmp(&y.1).unwrap())
        })
        .copied()
        .unwrap_or((best_err, 0.1, 1.0));

    Ok((alpha, l1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clip_eval_leaves_ordinary_scores_alone() {
        assert_eq!(clip_eval(0), 0);
        assert_eq!(clip_eval(250), 250);
        assert_eq!(clip_eval(-999), -999);
        assert_eq!(clip_eval(EVAL_CLIP_CP), EVAL_CLIP_CP);
    }

    #[test]
    fn test_clip_eval_bounds_mate_scores() {
        // Stockfish reports mates near +-10000; unclipped they dominate
        // the regression targets.
        assert_eq!(clip_eval(9999), EVAL_CLIP_CP);
        assert_eq!(clip_eval(-9999), -EVAL_CLIP_CP);
        assert_eq!(clip_eval(i32::MAX), EVAL_CLIP_CP);
        assert_eq!(clip_eval(i32::MIN), -EVAL_CLIP_CP);
    }

    #[test]
    fn test_clip_target_bounds_extreme_swings() {
        assert_eq!(clip_target(0.0), 0.0);
        assert_eq!(clip_target(250.0), 250.0);
        assert_eq!(clip_target(TARGET_CLIP_CP), TARGET_CLIP_CP);
        // A handful of huge swings would otherwise dominate a
        // squared-error fit.
        assert_eq!(clip_target(1200.0), TARGET_CLIP_CP);
        assert_eq!(clip_target(-1200.0), -TARGET_CLIP_CP);
        assert!(clip_target(f64::INFINITY).is_finite());
    }

    #[test]
    fn test_phase_threshold_scales_with_feature_count() {
        // The rule is rows-per-feature, so a wider model demands more
        // data before it earns a phase-specific fit.  Measured on a
        // 445-row sample, splitting three ways scored R2 0.130 against
        // the global model's 0.156.
        let required = |n_features: usize| MIN_ROWS_PER_PHASE_FEATURE * n_features;
        assert!(
            required(60) > 445 / 3,
            "a 60-feature model must demand more rows than a three-way \
             split of a 445-row sample provides"
        );
        assert!(required(60) > required(30), "wider models need more data");
    }

    #[test]
    fn test_classify_phase_thresholds() {
        // Phase value counts non-pawn, non-king pieces.
        let start = Chess::default();
        assert_eq!(board_phase_value(&start), 14);
        assert_eq!(classify_phase(&start), "opening");
    }
}
