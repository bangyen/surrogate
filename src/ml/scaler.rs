use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct StandardScaler {
    pub mean: Array1<f64>,
    pub scale: Array1<f64>,
    pub var: Array1<f64>,
    pub n_samples_seen: usize,
}

impl StandardScaler {
    pub fn new(n_features: usize) -> Self {
        StandardScaler {
            mean: Array1::zeros(n_features),
            scale: Array1::ones(n_features),
            var: Array1::zeros(n_features),
            n_samples_seen: 0,
        }
    }

    pub fn fit(&mut self, x: &Array2<f64>) {
        let n_samples = x.nrows();
        self.mean = x.mean_axis(Axis(0)).unwrap();
        self.var = x.var_axis(Axis(0), 0.0);
        self.scale = self.var.mapv(|v| if v > 0.0 { v.sqrt() } else { 1.0 });
        self.n_samples_seen = n_samples;
    }

    pub fn transform(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut x_scaled = x.clone();
        for mut row in x_scaled.rows_mut() {
            row -= &self.mean;
            row /= &self.scale;
        }
        x_scaled
    }

    pub fn inverse_transform(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut x_inv = x.clone();
        for mut row in x_inv.rows_mut() {
            row *= &self.scale;
            row += &self.mean;
        }
        x_inv
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_new_is_an_identity_transform() {
        let s = StandardScaler::new(2);
        let x = array![[3.0, -4.0]];
        assert_eq!(s.transform(&x), x);
    }

    #[test]
    fn test_fit_computes_mean_and_scale() {
        let mut s = StandardScaler::new(1);
        // mean 3, population variance 8/3 -> scale sqrt(8/3)
        s.fit(&array![[1.0], [3.0], [5.0]]);
        assert!((s.mean[0] - 3.0).abs() < 1e-12);
        assert!((s.var[0] - 8.0 / 3.0).abs() < 1e-12);
        assert!((s.scale[0] - (8.0f64 / 3.0).sqrt()).abs() < 1e-12);
        assert_eq!(s.n_samples_seen, 3);
    }

    #[test]
    fn test_fit_standardizes_to_zero_mean_unit_variance() {
        let mut s = StandardScaler::new(2);
        let x = array![[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]];
        s.fit(&x);
        let z = s.transform(&x);

        for col in 0..2 {
            let mean: f64 = z.column(col).iter().sum::<f64>() / 3.0;
            let var: f64 = z.column(col).iter().map(|v| v * v).sum::<f64>() / 3.0;
            assert!(mean.abs() < 1e-12, "column {col} mean was {mean}");
            assert!((var - 1.0).abs() < 1e-12, "column {col} var was {var}");
        }
    }

    #[test]
    fn test_constant_column_does_not_divide_by_zero() {
        let mut s = StandardScaler::new(1);
        s.fit(&array![[7.0], [7.0], [7.0]]);
        // Zero variance must fall back to a scale of 1, not 0.
        assert_eq!(s.scale[0], 1.0);
        let z = s.transform(&array![[7.0]]);
        assert!(z[[0, 0]].is_finite());
        assert_eq!(z[[0, 0]], 0.0);
    }

    #[test]
    fn test_inverse_transform_recovers_input() {
        let mut s = StandardScaler::new(2);
        let x = array![[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]];
        s.fit(&x);
        let recovered = s.inverse_transform(&s.transform(&x));
        for (a, b) in recovered.iter().zip(x.iter()) {
            assert!((a - b).abs() < 1e-10, "expected {b}, got {a}");
        }
    }

    #[test]
    fn test_json_round_trip() {
        let mut s = StandardScaler::new(2);
        s.fit(&array![[1.0, 10.0], [3.0, 20.0]]);
        let back: StandardScaler =
            serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
        assert_eq!(back.mean, s.mean);
        assert_eq!(back.scale, s.scale);
        assert_eq!(back.n_samples_seen, s.n_samples_seen);
    }
}
