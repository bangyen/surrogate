# Explainable Chess Engine (Rust Native)

A high-performance chess engine with integrated ML-driven move explanations, built entirely in Rust.

[![License](https://img.shields.io/github/license/bangyen/chess)](LICENSE)
[![CI](https://github.com/bangyen/chess/actions/workflows/ci.yml/badge.svg)](https://github.com/bangyen/chess/actions/workflows/ci.yml)

**A chess engine that explains its moves — and measures how honest those explanations are.**

<p align="center">
  <img src="docs/audit-demo.gif" alt="Demo preview" width="600">
</p>

## Quickstart

### Prerequisites

- [Rust 1.75+](https://rustup.rs/)
- [Just](https://github.com/casey/just) (optional, but recommended)
- [Stockfish Engine](https://stockfishchess.org/) (installed and in PATH, or set `STOCKFISH_PATH`)

### Installation

```bash
git clone https://github.com/bangyen/chess.git
cd chess
just build
```

A pretrained `model.json` is checked in, so the explanation commands work on
a fresh clone. Retrain it whenever the features or trainer change:

```bash
just train --n-positions 200
```

### Usage Options

**CLI Tools:**
```bash
# Measure explainability metrics and write audit-results.json
just metrics

# Verify a committed report still meets its targets
just metrics-check

# Inspect features and explanations for the starting position
just audit

# ...or on a specific position
just audit --fen "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1"

# Play interactive chess with explanations
just play

# Play a variant against the native engine (no Stockfish needed)
just variant koth
just variant antichess
```

**Web Interface:**
```bash
# Launch the Axum-based web dashboard
just web
# Then open http://localhost:5000
```

## Results

Measured by `just metrics` over 297 sampled positions at depth 12, against a
surrogate trained on 400 positions. Sampling is seeded, so the run is
reproducible: `just metrics` regenerates this table and `just metrics-check`
verifies the committed report.

| Metric | Value | Target | | n |
|--------|-------|--------|---|---|
| Explanation Sparsity | **2.67** | ≤ 4.0 | ✅ | 92 |
| Position Coverage | **1.000** | ≥ 0.70 | ✅ | 92 |
| Decisive Faithfulness | **0.738 ± 0.107** | ≥ 0.70 | ✅ | 65 |
| Move Ranking (τ) | **0.001** | *reported* | — | 297 |
| Fidelity (R²) | **0.015** | *reported* | — | 889 |

**What this says.** When the engine clearly prefers one move over another, the
surrogate agrees about three times in four, and it says so using fewer than
three features — an explanation a reader can actually follow. What it cannot
do is reproduce the engine's *full* ordering of moves, or predict how much an
evaluation will swing.

**Why τ and R² are reported rather than targeted.** Those two are not tuning
failures; they are the ceiling of this approach, established by measurement:

- With regularization at effectively zero, a linear model over these features
  explains only **R² ≈ 0.20 in-sample**, and 0.12–0.16 out of sample. The best
  single feature correlates 0.23 with the target; the median correlates 0.04.
- Move-ranking τ was **statistically indistinguishable from zero across four
  independently trained variants** (−0.05, 0.08, 0.13, 0.00).

Raising them means either much richer features or a nonlinear surrogate — and
a nonlinear model would forfeit the per-feature attribution that makes these
explanations readable in the first place. That trade is not worth making
silently, so the numbers stand as measured.

**A note on measurement power.** Faithfulness is computed only over positions
where the engine had a clear preference — 65 of 297 here — so its 95% interval
is ±0.107 even at this sample size. Differences smaller than that are not
detectable by this harness, which is why the targets above are set where
measurement can actually support them.

An earlier version of this table reported 86.7% faithfulness and R² 0.48.
Those came from a Python predecessor measuring a gradient-boosted model over
an enriched featureset, scored on a held-out split of its own training sample.
The measurement code did not survive the Rust rewrite; the numbers did. They
are not comparable to a linear model audited against freshly sampled
positions, and they have been replaced with what this code actually produces.

## Features

- **Feature Explainability Audit** — Native Rust implementation of move-ranking faithfulness metrics.
- **Interactive Chess Engine** — Play against Stockfish with real-time move explanations driven by a Rust-native surrogate model.
- **Axum Web Dashboard** — A modern, state-of-the-art web interface for position analysis and interactive gameplay.
- **Native ML Inference** — High-performance surrogate model implementation using `linfa` and `ndarray`, removing all Python dependencies.
- **Advanced Positional Analysis** — Sophisticated chess metrics including king safety, mobility, and piece activity.
- **Chess Variants** — The native engine also plays King of the Hill, Three-Check, and Antichess, each with its own evaluation. Explanations remain standard-chess only.

## Chess Variants

The search and move generation are shared with standard chess — every variant
implements the same position trait, so the alpha-beta search works unchanged.
What differs is *evaluation*: each variant wins differently, and reusing the
standard piece-square tables would produce legal but pointless play.

| Variant | Win condition | Evaluation |
|---------|---------------|------------|
| `koth` | March your king to a centre square | Standard eval plus a steep centre-proximity gradient |
| `3check` | Give check three times | Standard eval plus a term for checks remaining |
| `antichess` | Lose all your pieces; captures forced | Material inverted — pieces are a liability |

```bash
just variant --list      # show supported variants
just variant koth        # play King of the Hill
```

Atomic, Crazyhouse, Horde and Racing Kings are deliberately not offered: the
move generator supports them, but this engine has no evaluation for them and
would play legally while understanding nothing.

**Scope:** variants are engine-only. The ML explanation pipeline trains
against Stockfish, which plays standard chess exclusively, and the extracted
features encode standard-chess judgment. Explanations are therefore available
for standard chess alone.

## Repo Structure

```plaintext
chess/
├── src/
│   ├── engine/       # Native alpha-beta search, evaluation, SEE, Zobrist
│   │                 #   hashing, plus the Stockfish UCI interface
│   ├── features/     # High-performance feature extraction
│   ├── ml/           # Native ML model (Surrogate Model)
│   ├── web_server.rs # Axum web dashboard server
│   └── main.rs       # Unified CLI entry point
├── web/
│   ├── static/           # Front-end CSS/JS assets
│   └── templates/        # Tera templates for the dashboard
├── docs/                 # Documentation and design system
├── Cargo.toml            # Rust dependencies
└── justfile              # Orchestration targets
```

## Validation

- ✅ Continuous test coverage monitoring (`just test`)
- ✅ Zero-warning builds (`just lint`)
- ✅ Explainability metrics measured, not asserted (`just metrics`)
- ✅ Metric regressions fail the build (`just metrics-check`)

## License

This project is licensed under the [MIT License](LICENSE).
