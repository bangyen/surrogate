//! Explainable chess engine.
//!
//! The crate splits into a pure core -- move generation, search,
//! evaluation, feature extraction and ML inference -- and a native layer
//! that needs a subprocess, a filesystem or a socket.  Only the pure core
//! compiles for the browser, which is what the `wasm` build uses.

pub mod engine;
pub mod features;
pub mod ml;
pub mod pawn_cache;
pub mod variant;

pub use engine::eval;
pub use engine::search;
pub use engine::see;
pub use engine::zobrist;

// Native-only: these drive Stockfish, download tablebases, or serve HTTP.
#[cfg(feature = "native")]
pub mod audit;
#[cfg(feature = "native")]
pub mod syzygy;
#[cfg(feature = "native")]
pub mod web_server;

#[cfg(feature = "native")]
pub use syzygy::utils as syzygy_utils;

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub mod wasm_api;
