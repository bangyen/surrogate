use crate::ml::scaler::StandardScaler;
use ndarray::{Array1, ArrayView1};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PhaseModel {
    pub coefficients: Array1<f64>,
    pub intercept: f64,
    pub alpha: f64,
    pub l1_ratio: f64,
}

impl PhaseModel {
    pub fn predict(&self, features: &ArrayView1<f64>) -> f64 {
        self.coefficients.dot(features) + self.intercept
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PhaseEnsemble {
    pub feature_names: Vec<String>,
    pub phase_idx: i32,
    pub models: HashMap<String, PhaseModel>,
    pub global_model: Option<PhaseModel>,
    pub scaler: Option<StandardScaler>,
}

impl PhaseEnsemble {
    pub fn new(feature_names: Vec<String>) -> Self {
        let phase_idx = feature_names
            .iter()
            .position(|r| r == "phase")
            .map(|i| i as i32)
            .unwrap_or(-1);
        PhaseEnsemble {
            feature_names,
            phase_idx,
            models: HashMap::new(),
            global_model: None,
            scaler: None,
        }
    }

    pub fn get_phase(&self, features: &ArrayView1<f64>) -> String {
        if self.phase_idx == -1 {
            return "middlegame".to_string();
        }
        let p = features[self.phase_idx as usize];
        if p > 24.0 {
            "opening".to_string()
        } else if p > 12.0 {
            "middlegame".to_string()
        } else {
            "endgame".to_string()
        }
    }

    pub fn predict(&self, features: &ArrayView1<f64>) -> f64 {
        let phase = self.get_phase(features);
        let model = self.models.get(&phase).or(self.global_model.as_ref());

        match model {
            Some(m) => m.predict(features),
            None => 0.0,
        }
    }

    pub fn get_contributions(&self, features: &ArrayView1<f64>) -> Array1<f64> {
        let phase = self.get_phase(features);
        let model = self.models.get(&phase).or(self.global_model.as_ref());

        match model {
            Some(m) => {
                let mut contribs = m.coefficients.clone();
                for (i, val) in contribs.iter_mut().enumerate() {
                    *val *= features[i];
                }
                contribs
            }
            None => Array1::zeros(features.len()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn sample_model(coefficients: Array1<f64>, intercept: f64) -> PhaseModel {
        PhaseModel {
            coefficients,
            intercept,
            alpha: 0.1,
            l1_ratio: 0.5,
        }
    }

    #[test]
    fn test_phase_model_predict_is_linear() {
        let m = sample_model(array![2.0, -3.0], 1.5);
        let x = array![4.0, 2.0];
        // 2*4 + (-3)*2 + 1.5
        assert_eq!(m.predict(&x.view()), 3.5);
    }

    #[test]
    fn test_new_locates_phase_feature() {
        let e = PhaseEnsemble::new(vec![
            "material_diff".to_string(),
            "phase".to_string(),
            "mobility_us".to_string(),
        ]);
        assert_eq!(e.phase_idx, 1);

        // Absent "phase" is flagged with -1 rather than defaulting to 0.
        let e = PhaseEnsemble::new(vec!["material_diff".to_string()]);
        assert_eq!(e.phase_idx, -1);
    }

    #[test]
    fn test_get_phase_boundaries() {
        let e = PhaseEnsemble::new(vec!["phase".to_string()]);
        // Thresholds are exclusive: >24 opening, >12 middlegame, else endgame.
        assert_eq!(e.get_phase(&array![25.0].view()), "opening");
        assert_eq!(e.get_phase(&array![24.0].view()), "middlegame");
        assert_eq!(e.get_phase(&array![13.0].view()), "middlegame");
        assert_eq!(e.get_phase(&array![12.0].view()), "endgame");
        assert_eq!(e.get_phase(&array![0.0].view()), "endgame");
    }

    #[test]
    fn test_get_phase_defaults_without_phase_feature() {
        let e = PhaseEnsemble::new(vec!["material_diff".to_string()]);
        assert_eq!(e.get_phase(&array![99.0].view()), "middlegame");
    }

    #[test]
    fn test_predict_selects_phase_model_over_global() {
        let mut e = PhaseEnsemble::new(vec!["phase".to_string()]);
        e.global_model = Some(sample_model(array![0.0], 100.0));
        e.models
            .insert("endgame".to_string(), sample_model(array![0.0], 7.0));

        // An endgame position uses the endgame model...
        assert_eq!(e.predict(&array![5.0].view()), 7.0);
        // ...while a phase with no specific model falls back to global.
        assert_eq!(e.predict(&array![30.0].view()), 100.0);
    }

    #[test]
    fn test_predict_without_any_model_is_zero() {
        let e = PhaseEnsemble::new(vec!["phase".to_string()]);
        assert_eq!(e.predict(&array![20.0].view()), 0.0);
    }

    #[test]
    fn test_get_contributions_are_per_feature_products() {
        let mut e = PhaseEnsemble::new(vec!["phase".to_string(), "material_diff".to_string()]);
        e.global_model = Some(sample_model(array![1.0, -2.0], 50.0));

        let x = array![20.0, 3.0];
        let contribs = e.get_contributions(&x.view());
        // Contributions exclude the intercept.
        assert_eq!(contribs, array![20.0, -6.0]);
    }

    #[test]
    fn test_get_contributions_without_model_is_zeros() {
        let e = PhaseEnsemble::new(vec!["phase".to_string(), "material_diff".to_string()]);
        let contribs = e.get_contributions(&array![20.0, 3.0].view());
        assert_eq!(contribs, array![0.0, 0.0]);
    }

    #[test]
    fn test_ensemble_json_round_trip() {
        // model.json is written by `train` and read back by `audit`, so the
        // serialized form must survive a full round trip.
        let mut e = PhaseEnsemble::new(vec!["phase".to_string(), "material_diff".to_string()]);
        e.global_model = Some(sample_model(array![0.5, 2.0], -1.0));
        e.models
            .insert("opening".to_string(), sample_model(array![1.0, 1.0], 3.0));
        let mut scaler = StandardScaler::new(2);
        scaler.mean = array![1.0, 2.0];
        scaler.scale = array![2.0, 4.0];
        scaler.var = array![4.0, 16.0];
        scaler.n_samples_seen = 10;
        e.scaler = Some(scaler);

        let json = serde_json::to_string(&e).unwrap();
        let back: PhaseEnsemble = serde_json::from_str(&json).unwrap();

        assert_eq!(back.feature_names, e.feature_names);
        assert_eq!(back.phase_idx, e.phase_idx);
        assert_eq!(back.models.len(), 1);
        assert_eq!(back.models["opening"].intercept, 3.0);
        assert_eq!(
            back.global_model.as_ref().unwrap().coefficients,
            array![0.5, 2.0]
        );
        assert_eq!(back.scaler.as_ref().unwrap().scale, array![2.0, 4.0]);
        assert_eq!(back.scaler.as_ref().unwrap().n_samples_seen, 10);

        // Predictions must agree before and after the round trip.
        let x = array![30.0, 1.0];
        assert_eq!(back.predict(&x.view()), e.predict(&x.view()));
    }
}
