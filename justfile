RUN := "cargo run --bin chess-ai --"

# format code
fmt:
    cargo fmt --all

# check formatting
fmt-check:
    cargo fmt --all -- --check

# lint code
lint:
    cargo clippy -- -D warnings

# run tests
test *ARGS:
    cargo test {{ARGS}}

# run web dashboard (Rust-native)
web:
    @echo "Starting chess AI web dashboard (Rust Axum)..."
    {{RUN}} server

# build the project
build:
    cargo build --release

# train the surrogate model (Rust-native)
train *ARGS:
    {{RUN}} train {{ARGS}}

# run explainability audit (Rust-native)
audit *ARGS:
    {{RUN}} audit {{ARGS}}

# measure explainability metrics and write audit-results.json
metrics *ARGS:
    {{RUN}} metrics {{ARGS}}

# verify committed metrics still meet their targets
metrics-check:
    {{RUN}} metrics --check

# play an interactive game (Rust-native)
play:
    {{RUN}} play

# play a chess variant against the native engine
variant *ARGS:
    {{RUN}} variant {{ARGS}}

# download syzygy tablebases (3-5 piece)
syzygy-download dest="~/syzygy":
    {{RUN}} syzygy download --dest {{dest}}

# verify syzygy tablebase integration
syzygy-verify path="~/syzygy":
    {{RUN}} syzygy verify --syzygy-path {{path}}

# run all checks (fmt, lint, test)
check: fmt-check lint test

# run and verify everything
all: check
    @echo "All checks completed!"

# check environment dependencies
doctor:
    {{RUN}} doctor
