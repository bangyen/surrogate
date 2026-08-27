pub mod eval;
pub mod search;
pub mod see;
pub mod zobrist;

// Driving an external Stockfish process needs a subprocess and pipes, so
// it is native-only.  The search and evaluation above are pure and build
// for every target, including wasm.
#[cfg(feature = "native")]
pub mod uci;

#[cfg(feature = "native")]
pub use uci::{ExplainableEngine, UciEngine};
