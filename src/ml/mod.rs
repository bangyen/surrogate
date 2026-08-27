pub mod explainer;
pub mod model;
pub mod scaler;

// Training fits models with linfa against a live Stockfish, so it is
// native-only.  Inference -- which is what explanations need -- is pure.
#[cfg(feature = "native")]
pub mod trainer;

pub use explainer::SurrogateExplainer;
pub use model::{PhaseEnsemble, PhaseModel};
pub use scaler::StandardScaler;
#[cfg(feature = "native")]
pub use trainer::train_surrogate_model;
