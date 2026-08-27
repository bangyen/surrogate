//! Explainability audit.
//!
//! Measures how faithfully the surrogate model reproduces the engine's
//! own preferences, producing the metrics reported in the README.
//!
//! The feature vectors built here mirror `ml::trainer` exactly: absolute
//! features of the position after the move *and* the engine's best reply,
//! with the target being the evaluation swing across that pair.  A
//! mismatch would make surrogate predictions meaningless, so the two must
//! be kept in step.

pub mod metrics;

use anyhow::{anyhow, Result};
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use shakmaty::{Chess, Position};

use crate::engine::UciEngine;
use crate::features::extract_features;
use crate::ml::trainer::{clip_eval, clip_target, generate_stratified_positions_seeded};
use crate::ml::PhaseEnsemble;

/// Thresholds each metric is expected to meet.
///
/// These are regression guards, not aspirations: each sits below the
/// measured baseline with enough margin to absorb sampling noise, so a
/// failure means something actually broke rather than that the approach
/// fell short of an ideal.
pub const TARGET_FAITHFULNESS: f64 = 0.80;
pub const TARGET_SPARSITY: f64 = 5.0;
pub const TARGET_COVERAGE: f64 = 0.70;

/// A target no measurement can fail, for metrics worth showing but not
/// gating.  Infinity is not representable in JSON, so it is finite.
pub const REPORTED_ONLY: f64 = -1.0e9;

pub const TARGET_TAU: f64 = 0.15;
pub const TARGET_R2: f64 = 0.40;

/// Centipawn gap below which a position's top two moves are considered
/// too close to tell apart, and so excluded from faithfulness.
pub const DEFAULT_GAP_CP: f64 = 50.0;

/// Minimum absolute coefficient for a feature to count toward coverage.
pub const DEFAULT_WEIGHT_THRESHOLD: f64 = 0.01;

/// Default sampling seed, fixed so repeated audits are comparable.
pub const DEFAULT_SEED: u64 = 0x5EED_C4E5;

/// Centipawn gap between the two moves' *outcomes* required to call the
/// comparison decisive.  Distinct from `DEFAULT_GAP_CP`, which filters on
/// the engine's scores before either move is played.
const DECISIVE_SWING_CP: f64 = 80.0;

/// Knobs for a single audit run.
#[derive(Clone, Debug)]
pub struct AuditConfig {
    pub stockfish_path: String,
    pub n_positions: usize,
    pub depth: u32,
    pub multipv: u32,
    pub gap_cp: f64,
    pub weight_threshold: f64,
    /// Seed for position sampling, so runs are reproducible and two
    /// models can be compared on identical positions.
    pub seed: u64,
}

impl Default for AuditConfig {
    fn default() -> Self {
        AuditConfig {
            stockfish_path: "stockfish".to_string(),
            n_positions: 100,
            depth: 12,
            multipv: 3,
            gap_cp: DEFAULT_GAP_CP,
            weight_threshold: DEFAULT_WEIGHT_THRESHOLD,
            seed: DEFAULT_SEED,
        }
    }
}

/// One metric's measured value against the target it is held to.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Metric {
    pub name: String,
    pub value: f64,
    pub target: f64,
    /// True when higher is better; false when the target is a ceiling.
    pub higher_is_better: bool,
    /// How many positions contributed to this value.
    pub n: usize,
}

impl Metric {
    /// Whether this metric is measured and reported but not gated.
    pub fn is_reported_only(&self) -> bool {
        self.target <= REPORTED_ONLY
    }

    pub fn passes(&self) -> bool {
        if self.is_reported_only() {
            return true;
        }
        if self.higher_is_better {
            self.value >= self.target
        } else {
            self.value <= self.target
        }
    }
}

/// A complete audit run, suitable for serializing and comparing over time.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AuditReport {
    pub metrics: Vec<Metric>,
    pub n_positions_requested: usize,
    pub n_positions_evaluated: usize,
    pub depth: u32,
    /// Sampling seed, so a report identifies the positions it measured.
    #[serde(default)]
    pub seed: u64,
}

impl AuditReport {
    /// True when every metric meets its target.
    pub fn passes(&self) -> bool {
        self.metrics.iter().all(|m| m.passes())
    }

    /// Metrics that fall short of their target.
    pub fn failures(&self) -> Vec<&Metric> {
        self.metrics.iter().filter(|m| !m.passes()).collect()
    }

    /// Render the report as the README's results table.
    pub fn to_markdown(&self) -> String {
        let mut out = String::from("| Metric | Value | Target |\n|--------|-------|--------|\n");
        for m in &self.metrics {
            let target = if m.is_reported_only() {
                "*reported*".to_string()
            } else {
                let comparator = if m.higher_is_better { "≥" } else { "≤" };
                format!("{} {:.2}", comparator, m.target)
            };
            out.push_str(&format!(
                "| {} | **{:.3}** | {} |\n",
                m.name, m.value, target
            ));
        }
        out
    }
}

/// Running tallies collected across audited positions.
#[derive(Default)]
struct Tallies {
    decisive_hits: usize,
    decisive_total: usize,
    coverage_hits: usize,
    coverage_total: usize,
    sparsity_counts: Vec<usize>,
    taus: Vec<f64>,
    /// Observed and predicted values, grouped by position, so fidelity
    /// can be measured on the axis the surrogate actually models.
    groups: Vec<(Vec<f64>, Vec<f64>)>,
}

/// Play `mv` followed by the engine's best reply, returning the
/// evaluation swing and the resulting feature vector.
///
/// This mirrors `ml::trainer`'s construction; see the module docs.
fn evaluate_move(
    engine: &mut UciEngine,
    pos: &Chess,
    mv_uci: &str,
    base_eval: i32,
    feature_names: &[String],
    depth: u32,
) -> Option<(f64, Array1<f64>)> {
    let uci: shakmaty::uci::UciMove = mv_uci.parse().ok()?;
    let m = uci.to_move(pos).ok()?;

    let mut after = pos.clone();
    after.play_unchecked(m);

    let fen_after =
        shakmaty::fen::Fen::from_position(&after, shakmaty::EnPassantMode::Always).to_string();
    let reply_uci = engine.get_best_move(&fen_after, depth).ok()?;
    let reply: shakmaty::uci::UciMove = reply_uci.parse().ok()?;
    let rm = reply.to_move(&after).ok()?;
    after.play_unchecked(rm);

    let fen_final =
        shakmaty::fen::Fen::from_position(&after, shakmaty::EnPassantMode::Always).to_string();
    let after_eval = engine.get_evaluation(&fen_final, depth).ok()?;
    // Clipped exactly as ml::trainer clips its targets; measuring against
    // an unclipped swing would compare the model to a different scale.
    let delta = clip_target((clip_eval(after_eval) - clip_eval(base_eval)) as f64);

    let feats = extract_features(&after);
    let mut vec = Array1::zeros(feature_names.len());
    for (i, name) in feature_names.iter().enumerate() {
        vec[i] = *feats.get(name).unwrap_or(&0.0) as f64;
    }

    Some((delta, vec))
}

/// Apply the model's scaler, matching how the model was trained.
fn scale(model: &PhaseEnsemble, raw: &Array1<f64>) -> Array1<f64> {
    match &model.scaler {
        Some(s) => {
            let mut v = raw.clone();
            v -= &s.mean;
            v /= &s.scale;
            v
        }
        None => raw.clone(),
    }
}

/// Coefficients of whichever phase model applies to this position.
///
/// The metric definitions assume one coefficient vector; with a phase
/// ensemble we use the model that would actually produce the explanation.
fn coefficients_for(model: &PhaseEnsemble, scaled: &Array1<f64>) -> Option<Array1<f64>> {
    let phase = model.get_phase(&scaled.view());
    model
        .models
        .get(&phase)
        .or(model.global_model.as_ref())
        .map(|m| m.coefficients.clone())
}

/// Run the audit, collecting metrics over freshly sampled positions.
pub fn run_audit(model: &PhaseEnsemble, cfg: &AuditConfig) -> Result<AuditReport> {
    if model.feature_names.is_empty() {
        return Err(anyhow!("model has no features; train a model first"));
    }

    let mut engine = UciEngine::new(&cfg.stockfish_path)?;
    let boards = generate_stratified_positions_seeded(cfg.n_positions, cfg.seed);
    let names = &model.feature_names;

    let mut t = Tallies::default();
    let mut evaluated = 0usize;

    for (i, pos) in boards.iter().enumerate() {
        if (i + 1) % 10 == 0 {
            println!("Auditing position {}/{}...", i + 1, boards.len());
        }

        let fen =
            shakmaty::fen::Fen::from_position(pos, shakmaty::EnPassantMode::Always).to_string();

        let Ok(base_eval) = engine.get_evaluation(&fen, cfg.depth) else {
            continue;
        };
        let Ok(candidates) = engine.get_top_moves(&fen, cfg.depth, cfg.multipv) else {
            continue;
        };
        if candidates.len() < 2 {
            continue;
        }

        let mut sorted = candidates.clone();
        sorted.sort_by_key(|a| std::cmp::Reverse(a.1));

        // Evaluate every candidate once and keep the results: the
        // faithfulness metrics below need the top two again, and each
        // evaluation costs two more engine searches.
        let mut evaluated_moves = Vec::new();
        let mut sf_scores = Vec::new();
        let mut sur_scores = Vec::new();
        let mut group_true: Vec<f64> = Vec::new();
        let mut group_pred: Vec<f64> = Vec::new();
        for (mv, _) in &sorted {
            let Some((delta, vec)) =
                evaluate_move(&mut engine, pos, mv, base_eval, names, cfg.depth)
            else {
                evaluated_moves.push(None);
                continue;
            };

            let scaled = scale(model, &vec);
            let prediction = model.predict(&scaled.view());
            sf_scores.push(delta);
            sur_scores.push(prediction);
            group_true.push(delta);
            group_pred.push(prediction);
            evaluated_moves.push(Some((delta, scaled)));
        }
        if sf_scores.len() >= 2 {
            t.taus.push(metrics::kendall_tau(&sf_scores, &sur_scores));
            t.groups.push((group_true, group_pred));
            evaluated += 1;
        }

        // Faithfulness, sparsity and coverage compare the top two moves,
        // but only when the engine clearly preferred one of them.
        let (_, best_cp) = &sorted[0];
        let (_, second_cp) = &sorted[1];
        if (best_cp - second_cp).abs() < cfg.gap_cp as i32 {
            continue;
        }

        let (Some(Some((delta_best, scaled_best))), Some(Some((delta_second, scaled_second)))) =
            (evaluated_moves.first(), evaluated_moves.get(1))
        else {
            continue;
        };

        let sur_best = model.predict(&scaled_best.view());
        let sur_second = model.predict(&scaled_second.view());

        if (delta_best - delta_second).abs() >= DECISIVE_SWING_CP {
            t.decisive_total += 1;
            if (sur_best > sur_second) == (delta_best > delta_second) {
                t.decisive_hits += 1;
            }
        }

        let contributions = model.get_contributions(&scaled_best.view());
        let contrib: Vec<f64> = contributions.to_vec();

        if let Some(sp) = metrics::sparsity_count(&contrib) {
            t.sparsity_counts.push(sp);
        }

        if let Some(coef) = coefficients_for(model, scaled_best) {
            t.coverage_total += 1;
            if metrics::is_covered(&coef.to_vec(), &contrib, cfg.weight_threshold) {
                t.coverage_hits += 1;
            }
        }
    }

    Ok(build_report(&t, cfg, evaluated))
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn ratio(hits: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        hits as f64 / total as f64
    }
}

fn build_report(t: &Tallies, cfg: &AuditConfig, evaluated: usize) -> AuditReport {
    let sparsity_mean = mean(
        &t.sparsity_counts
            .iter()
            .map(|c| *c as f64)
            .collect::<Vec<_>>(),
    );

    AuditReport {
        metrics: vec![
            Metric {
                name: "Decisive Faithfulness".to_string(),
                value: ratio(t.decisive_hits, t.decisive_total),
                target: TARGET_FAITHFULNESS,
                higher_is_better: true,
                n: t.decisive_total,
            },
            Metric {
                name: "Explanation Sparsity".to_string(),
                value: sparsity_mean,
                target: TARGET_SPARSITY,
                higher_is_better: false,
                n: t.sparsity_counts.len(),
            },
            Metric {
                name: "Position Coverage".to_string(),
                value: ratio(t.coverage_hits, t.coverage_total),
                target: TARGET_COVERAGE,
                higher_is_better: true,
                n: t.coverage_total,
            },
            Metric {
                name: "Move Ranking (tau)".to_string(),
                value: mean(&t.taus),
                target: TARGET_TAU,
                higher_is_better: true,
                n: t.taus.len(),
            },
            Metric {
                name: "Fidelity (delta-R2)".to_string(),
                value: metrics::within_group_r2(&t.groups),
                target: TARGET_R2,
                higher_is_better: true,
                n: t.groups.iter().map(|(a, _)| a.len()).sum(),
            },
        ],
        n_positions_requested: cfg.n_positions,
        n_positions_evaluated: evaluated,
        depth: cfg.depth,
        seed: cfg.seed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml::PhaseModel;

    fn metric(value: f64, target: f64, higher_is_better: bool) -> Metric {
        Metric {
            name: "test".to_string(),
            value,
            target,
            higher_is_better,
            n: 10,
        }
    }

    #[test]
    fn test_reported_only_metrics_never_gate() {
        // Tau and fidelity are measured and shown but not gated, since
        // they sit at the ceiling of a linear surrogate.
        let m = Metric {
            name: "Move Ranking (tau)".to_string(),
            value: -0.9,
            target: REPORTED_ONLY,
            higher_is_better: true,
            n: 100,
        };
        assert!(m.is_reported_only());
        assert!(
            m.passes(),
            "a reported-only metric must never fail the gate"
        );

        // An ordinary target is still enforced.
        assert!(!metric(0.1, 0.7, true).is_reported_only());
        assert!(!metric(0.1, 0.7, true).passes());
    }

    #[test]
    fn test_reported_only_metrics_render_without_a_threshold() {
        let report = AuditReport {
            metrics: vec![Metric {
                name: "Fidelity (R2)".to_string(),
                value: 0.015,
                target: REPORTED_ONLY,
                higher_is_better: true,
                n: 889,
            }],
            n_positions_requested: 300,
            n_positions_evaluated: 297,
            depth: 12,
            seed: DEFAULT_SEED,
        };
        let md = report.to_markdown();
        assert!(md.contains("*reported*"), "got: {md}");
        assert!(
            !md.contains("-1000000000"),
            "sentinel leaked into output: {md}"
        );
    }

    #[test]
    fn test_metric_passes_when_above_a_floor() {
        assert!(metric(0.9, 0.8, true).passes());
        assert!(metric(0.8, 0.8, true).passes(), "the target itself passes");
        assert!(!metric(0.79, 0.8, true).passes());
    }

    #[test]
    fn test_metric_passes_when_below_a_ceiling() {
        // Sparsity is better when smaller, so the comparison inverts.
        assert!(metric(2.5, 4.0, false).passes());
        assert!(metric(4.0, 4.0, false).passes());
        assert!(!metric(4.1, 4.0, false).passes());
    }

    #[test]
    fn test_report_passes_only_when_every_metric_does() {
        let mut report = AuditReport {
            metrics: vec![metric(0.9, 0.8, true), metric(2.0, 4.0, false)],
            n_positions_requested: 10,
            n_positions_evaluated: 10,
            depth: 12,
            seed: DEFAULT_SEED,
        };
        assert!(report.passes());
        assert!(report.failures().is_empty());

        report.metrics.push(metric(0.1, 0.5, true));
        assert!(!report.passes(), "one failing metric fails the report");
        assert_eq!(report.failures().len(), 1);
    }

    #[test]
    fn test_report_markdown_uses_the_right_comparator() {
        let report = AuditReport {
            metrics: vec![
                Metric {
                    name: "Decisive Faithfulness".to_string(),
                    value: 0.867,
                    target: 0.8,
                    higher_is_better: true,
                    n: 30,
                },
                Metric {
                    name: "Explanation Sparsity".to_string(),
                    value: 2.5,
                    target: 4.0,
                    higher_is_better: false,
                    n: 30,
                },
            ],
            n_positions_requested: 100,
            n_positions_evaluated: 90,
            depth: 12,
            seed: DEFAULT_SEED,
        };

        let md = report.to_markdown();
        assert!(md.contains("| Metric | Value | Target |"));
        assert!(md.contains("**0.867**"), "got: {md}");
        assert!(md.contains("≥ 0.80"), "floors use >=: {md}");
        assert!(md.contains("≤ 4.00"), "ceilings use <=: {md}");
    }

    #[test]
    fn test_report_json_round_trip() {
        // The report is committed to disk and re-read by `--check`.
        let report = AuditReport {
            metrics: vec![metric(0.9, 0.8, true)],
            n_positions_requested: 100,
            n_positions_evaluated: 88,
            depth: 12,
            seed: DEFAULT_SEED,
        };
        let json = serde_json::to_string_pretty(&report).unwrap();
        let back: AuditReport = serde_json::from_str(&json).unwrap();

        assert_eq!(back.n_positions_evaluated, 88);
        assert_eq!(back.metrics.len(), 1);
        assert_eq!(back.metrics[0].value, 0.9);
        assert_eq!(back.passes(), report.passes());
    }

    #[test]
    fn test_run_audit_rejects_an_untrained_model() {
        // Without features there is nothing to measure; fail loudly
        // rather than reporting zeros as if they were real.
        let model = PhaseEnsemble::new(vec![]);
        let err = run_audit(&model, &AuditConfig::default()).unwrap_err();
        assert!(
            err.to_string().contains("no features"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_empty_tallies_produce_zeroed_metrics() {
        // A run that evaluated nothing must not report NaN or claim success.
        let report = build_report(&Tallies::default(), &AuditConfig::default(), 0);
        for m in &report.metrics {
            assert!(m.value.is_finite(), "{} was not finite", m.name);
            assert_eq!(m.n, 0);
        }
        assert!(!report.passes(), "an empty run must not pass");
    }

    #[test]
    fn test_coefficients_follow_the_phase() {
        let mut model = PhaseEnsemble::new(vec!["phase".to_string()]);
        model.global_model = Some(PhaseModel {
            coefficients: Array1::from(vec![1.0]),
            intercept: 0.0,
            alpha: 0.1,
            l1_ratio: 0.5,
        });
        model.models.insert(
            "endgame".to_string(),
            PhaseModel {
                coefficients: Array1::from(vec![9.0]),
                intercept: 0.0,
                alpha: 0.1,
                l1_ratio: 0.5,
            },
        );

        // An endgame position uses the endgame coefficients...
        let endgame = Array1::from(vec![5.0]);
        assert_eq!(coefficients_for(&model, &endgame).unwrap()[0], 9.0);
        // ...and a phase without its own model falls back to global.
        let opening = Array1::from(vec![30.0]);
        assert_eq!(coefficients_for(&model, &opening).unwrap()[0], 1.0);
    }

    #[test]
    fn test_scale_applies_the_models_scaler() {
        let mut model = PhaseEnsemble::new(vec!["a".to_string()]);
        let mut scaler = crate::ml::StandardScaler::new(1);
        scaler.mean = Array1::from(vec![10.0]);
        scaler.scale = Array1::from(vec![2.0]);
        model.scaler = Some(scaler);

        let scaled = scale(&model, &Array1::from(vec![30.0]));
        assert_eq!(scaled[0], 10.0, "(30 - 10) / 2");

        // With no scaler the vector passes through untouched.
        let bare = PhaseEnsemble::new(vec!["a".to_string()]);
        assert_eq!(scale(&bare, &Array1::from(vec![30.0]))[0], 30.0);
    }
}
