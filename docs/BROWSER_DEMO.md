# Browser demo

The repository has a local browser demo in `web/demo.html`.

## Run locally

Install Rust via rustup, [Just](https://github.com/casey/just),
[wasm-pack](https://rustwasm.github.io/wasm-pack/installer/), and Python 3.
The repository pins Rust and includes the `wasm32-unknown-unknown` target.
Ensure rustup's binaries take precedence over a separate Rust installation
(for example, Homebrew Rust): `export PATH="$HOME/.cargo/bin:$PATH"`.
From the repository root:

```bash
just wasm
just demo
```

Open <http://localhost:8000/demo.html>. The engine and explanation model
run in the browser; Stockfish is not required. Serve the files over HTTP
rather than opening the HTML as a local file, so ES modules and the model
fetch can load.
