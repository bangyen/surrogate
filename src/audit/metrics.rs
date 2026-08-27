//! Explainability metrics.
//!
//! These are ports of the definitions used by this project's original
//! Python audit, kept faithful so the reported numbers mean the same
//! thing they always did.  The surrogate model differs (a linear
//! phase ensemble here, a gradient-boosted tree there), so the values
//! themselves are not expected to match the historical ones.

/// Kendall tau-b correlation between two rankings.
///
/// Pairs are compared by the sign of their ordering in each ranking:
/// agreeing pairs are concordant, disagreeing pairs discordant.  Returns
/// 0.0 for rankings too short to contain a pair.
pub fn kendall_tau(rank_a: &[f64], rank_b: &[f64]) -> f64 {
    assert_eq!(
        rank_a.len(),
        rank_b.len(),
        "rankings must have equal length, got {} and {}",
        rank_a.len(),
        rank_b.len()
    );

    let n = rank_a.len();
    let mut concordant = 0i64;
    let mut discordant = 0i64;

    for i in 0..n {
        for j in (i + 1)..n {
            let da = sign(rank_a[i], rank_a[j]);
            let db = sign(rank_b[i], rank_b[j]);
            if da == db {
                concordant += 1;
            } else {
                discordant += 1;
            }
        }
    }

    let denom = concordant + discordant;
    if denom > 0 {
        (concordant - discordant) as f64 / denom as f64
    } else {
        0.0
    }
}

/// `(a > b) - (a < b)`, matching the original implementation's tie handling.
fn sign(a: f64, b: f64) -> i32 {
    i32::from(a > b) - i32::from(a < b)
}

/// Coefficient of determination between observed and predicted values.
///
/// This is the "Fidelity" metric: how much of the engine's evaluation
/// swing the surrogate accounts for.  Returns 0.0 when the observations
/// carry no variance, since R² is undefined there.
pub fn r2_score(y_true: &[f64], y_pred: &[f64]) -> f64 {
    assert_eq!(
        y_true.len(),
        y_pred.len(),
        "inputs must have equal length, got {} and {}",
        y_true.len(),
        y_pred.len()
    );

    if y_true.is_empty() {
        return 0.0;
    }

    let mean = y_true.iter().sum::<f64>() / y_true.len() as f64;
    let ss_tot: f64 = y_true.iter().map(|y| (y - mean).powi(2)).sum();
    let ss_res: f64 = y_true
        .iter()
        .zip(y_pred)
        .map(|(y, p)| (y - p).powi(2))
        .sum();

    if ss_tot.abs() < 1e-12 {
        return 0.0;
    }
    1.0 - ss_res / ss_tot
}

/// Coefficient of determination measured *within* each group.
///
/// The surrogate is fit to the differences between candidate moves from
/// one position, not to absolute evaluation levels, so scoring it on
/// levels would measure something it never claimed to model.  Each
/// group is centred on its own mean before the comparison, which is the
/// same transformation the trainer applies.
///
/// Returns 0.0 when no group carries any within-group variance.
pub fn within_group_r2(groups: &[(Vec<f64>, Vec<f64>)]) -> f64 {
    let mut ss_tot = 0.0;
    let mut ss_res = 0.0;

    for (observed, predicted) in groups {
        if observed.len() < 2 || observed.len() != predicted.len() {
            continue;
        }
        let n = observed.len() as f64;
        let obs_mean = observed.iter().sum::<f64>() / n;
        let pred_mean = predicted.iter().sum::<f64>() / n;

        for (o, p) in observed.iter().zip(predicted) {
            let centred_obs = o - obs_mean;
            let centred_pred = p - pred_mean;
            ss_tot += centred_obs.powi(2);
            ss_res += (centred_obs - centred_pred).powi(2);
        }
    }

    if ss_tot.abs() < 1e-12 {
        return 0.0;
    }
    1.0 - ss_res / ss_tot
}

/// Number of features needed to reach 80% of the total absolute
/// contribution — the "sparsity" of an explanation.
///
/// A small number means a few features carry the explanation, which is
/// what makes it human-readable.  Returns `None` when contributions are
/// all essentially zero and there is nothing to explain.
pub fn sparsity_count(contributions: &[f64]) -> Option<usize> {
    let total: f64 = contributions.iter().map(|c| c.abs()).sum();
    if total <= 1e-9 {
        return None;
    }

    let mut sorted: Vec<f64> = contributions.iter().map(|c| c.abs()).collect();
    sorted.sort_by(|a, b| b.partial_cmp(a).expect("contributions must not be NaN"));

    let mut cumulative = 0.0;
    for (i, v) in sorted.iter().enumerate() {
        cumulative += v;
        if cumulative >= 0.8 * total {
            return Some(i + 1);
        }
    }
    Some(sorted.len())
}

/// Whether a position is "covered": at least two features with a
/// meaningful coefficient actually contribute to the explanation.
///
/// A position explained by a single feature is fragile, so coverage
/// tracks how often the surrogate has more than one thing to say.
pub fn is_covered(coefficients: &[f64], contributions: &[f64], weight_threshold: f64) -> bool {
    let strong = coefficients
        .iter()
        .zip(contributions)
        .filter(|(c, contrib)| c.abs() >= weight_threshold && contrib.abs() > 0.0)
        .count();
    strong >= 2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kendall_tau_perfect_agreement() {
        assert_eq!(kendall_tau(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]), 1.0);
        // Agreement is about order, not magnitude.
        assert_eq!(kendall_tau(&[1.0, 2.0, 3.0], &[10.0, 50.0, 99.0]), 1.0);
    }

    #[test]
    fn test_kendall_tau_perfect_disagreement() {
        assert_eq!(kendall_tau(&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0]), -1.0);
    }

    #[test]
    fn test_kendall_tau_partial_agreement() {
        // Three pairs; swapping the last two elements makes one discordant.
        let tau = kendall_tau(&[1.0, 2.0, 3.0], &[1.0, 3.0, 2.0]);
        assert!(
            (tau - 1.0 / 3.0).abs() < 1e-12,
            "expected 1/3 (2 concordant, 1 discordant), got {tau}"
        );
    }

    #[test]
    fn test_kendall_tau_degenerate_inputs() {
        // Fewer than two elements means no pairs, so no correlation.
        assert_eq!(kendall_tau(&[], &[]), 0.0);
        assert_eq!(kendall_tau(&[1.0], &[5.0]), 0.0);
    }

    #[test]
    fn test_kendall_tau_treats_ties_as_discordant() {
        // Matches the original: a tie in one ranking but not the other
        // yields differing signs, counted as discordant.
        assert_eq!(kendall_tau(&[1.0, 1.0], &[1.0, 2.0]), -1.0);
        // Ties on both sides agree.
        assert_eq!(kendall_tau(&[1.0, 1.0], &[3.0, 3.0]), 1.0);
    }

    #[test]
    #[should_panic(expected = "equal length")]
    fn test_kendall_tau_rejects_mismatched_lengths() {
        kendall_tau(&[1.0, 2.0], &[1.0]);
    }

    #[test]
    fn test_r2_score_perfect_prediction() {
        assert_eq!(r2_score(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]), 1.0);
    }

    #[test]
    fn test_r2_score_mean_prediction_is_zero() {
        // Predicting the mean every time explains none of the variance.
        let y = [1.0, 2.0, 3.0];
        assert!(r2_score(&y, &[2.0, 2.0, 2.0]).abs() < 1e-12);
    }

    #[test]
    fn test_r2_score_can_go_negative() {
        // Worse than predicting the mean.
        let r2 = r2_score(&[1.0, 2.0, 3.0], &[10.0, -5.0, 20.0]);
        assert!(r2 < 0.0, "expected negative R², got {r2}");
    }

    #[test]
    fn test_r2_score_degenerate_inputs() {
        assert_eq!(r2_score(&[], &[]), 0.0);
        // Zero variance leaves R² undefined; report 0 rather than NaN.
        let r2 = r2_score(&[5.0, 5.0, 5.0], &[5.0, 5.0, 5.0]);
        assert!(r2.is_finite(), "R² must not be NaN");
        assert_eq!(r2, 0.0);
    }

    #[test]
    fn test_within_group_r2_ignores_group_level_offsets() {
        // Predictions that track the within-group pattern perfectly but
        // sit at a wholly different level must still score 1.0: the
        // surrogate models differences, not levels.
        let groups = vec![
            (vec![10.0, 20.0, 30.0], vec![1010.0, 1020.0, 1030.0]),
            (vec![5.0, 15.0], vec![-995.0, -985.0]),
        ];
        assert!((within_group_r2(&groups) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_within_group_r2_penalises_wrong_ordering() {
        // Correct level, reversed within-group pattern.
        let groups = vec![(vec![10.0, 20.0, 30.0], vec![30.0, 20.0, 10.0])];
        assert!(
            within_group_r2(&groups) < 0.0,
            "an inverted ordering must score below zero"
        );
    }

    #[test]
    fn test_within_group_r2_degenerate_inputs() {
        assert_eq!(within_group_r2(&[]), 0.0);
        // Single-element groups carry no within-group variance.
        assert_eq!(within_group_r2(&[(vec![5.0], vec![9.0])]), 0.0);
        // A group where every observation is identical contributes none.
        assert_eq!(within_group_r2(&[(vec![7.0, 7.0], vec![1.0, 2.0])]), 0.0);
        // Mismatched lengths are skipped rather than panicking.
        assert_eq!(within_group_r2(&[(vec![1.0, 2.0], vec![1.0])]), 0.0);
    }

    #[test]
    fn test_sparsity_counts_features_to_reach_80_percent() {
        // One feature carries everything.
        assert_eq!(sparsity_count(&[100.0, 0.0, 0.0]), Some(1));
        // Four equal features: three are needed to pass 80%.
        assert_eq!(sparsity_count(&[10.0, 10.0, 10.0, 10.0]), Some(4));
        // 60 + 30 = 90% of 100, so two suffice.
        assert_eq!(sparsity_count(&[60.0, 30.0, 5.0, 5.0]), Some(2));
    }

    #[test]
    fn test_sparsity_uses_absolute_values() {
        // Sign should not change how concentrated an explanation is.
        assert_eq!(
            sparsity_count(&[-60.0, 30.0, -5.0, 5.0]),
            sparsity_count(&[60.0, 30.0, 5.0, 5.0])
        );
    }

    #[test]
    fn test_sparsity_none_when_nothing_contributes() {
        assert_eq!(sparsity_count(&[]), None);
        assert_eq!(sparsity_count(&[0.0, 0.0]), None);
        // Below the 1e-9 floor counts as nothing to explain.
        assert_eq!(sparsity_count(&[1e-12, -1e-12]), None);
    }

    #[test]
    fn test_coverage_requires_two_strong_features() {
        // Two coefficients above threshold, both contributing.
        assert!(is_covered(&[1.0, 1.0], &[5.0, 5.0], 0.5));
        // Only one strong feature is too fragile to count as covered.
        assert!(!is_covered(&[1.0, 0.1], &[5.0, 5.0], 0.5));
        // Strong coefficients that contribute nothing do not count.
        assert!(!is_covered(&[1.0, 1.0], &[5.0, 0.0], 0.5));
        assert!(!is_covered(&[], &[], 0.5));
    }

    #[test]
    fn test_coverage_uses_absolute_magnitudes() {
        // Negative coefficients are just as informative as positive ones.
        assert!(is_covered(&[-1.0, -1.0], &[-5.0, -5.0], 0.5));
    }
}
