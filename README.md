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

Measured over 98 sampled positions at depth 12 by `just metrics`, against a
surrogate model trained on 200 positions. Regenerate with `just metrics`;
verify a committed report with `just metrics-check`.

| Metric | Value | Target | |
|--------|-------|--------|---|
| Decisive Faithfulness | **0.800** | ≥ 0.80 | ✅ |
| Position Coverage | **1.000** | ≥ 0.70 | ✅ |
| Explanation Sparsity | **9.49** | ≤ 4.0 | ❌ |
| Move Ranking (τ) | **0.299** | ≥ 0.45 | ❌ |
| Fidelity (R²) | **-0.015** | ≥ 0.35 | ❌ |

**Reading these honestly:** the surrogate agrees with the engine about which
of two clearly-separated moves is better 80% of the time, and almost always
has more than one feature to point at. It is *not* yet a faithful model of
the engine's evaluation: an R² near zero means it predicts about as well as
guessing the mean, and explanations currently lean on ~9 features where a
readable one would use 3–4.

These numbers replace an earlier table that reported 86.7% faithfulness and
R² 0.48. That table was inherited from a Python predecessor whose measurement
code did not survive the Rust rewrite — it described a gradient-boosted model
over an enriched featureset, scored on a held-out split of its own training
sample. The current pipeline is a linear model over plain features, audited
against freshly sampled positions, so the two are not comparable. The
numbers above are what this code actually produces.

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
