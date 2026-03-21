use anyhow::Result;
use axum::{
    extract::State,
    http::StatusCode,
    response::{Html, IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use shakmaty::{Chess, Position};
use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};
use tera::{Context, Tera};
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;

use crate::engine::ExplainableEngine;
use crate::features::extract_features;
use crate::ml::{train_surrogate_model, PhaseEnsemble, SurrogateExplainer};

#[derive(Clone)]
pub struct GameState {
    pub board: Chess,
    pub engine: Option<Arc<RwLock<ExplainableEngine>>>,
    pub model: Option<PhaseEnsemble>,
    pub stockfish_path: String,
    pub model_ready: bool,
    pub training_error: bool,
    pub history: Vec<String>,
}

impl GameState {
    pub fn new(stockfish_path: String) -> Self {
        GameState {
            board: Chess::default(),
            engine: None,
            model: None,
            stockfish_path,
            model_ready: false,
            training_error: false,
            history: Vec::new(),
        }
    }

    pub fn reset(&mut self) {
        self.board = Chess::default();
        self.history.clear();
    }
}

type SharedState = Arc<RwLock<GameState>>;

#[derive(Serialize)]
struct BoardState {
    fen: String,
    legal_moves: Vec<String>,
    is_game_over: bool,
    result: Option<String>,
    turn: String,
}

#[derive(Deserialize)]
struct MoveRequest {
    #[serde(rename = "move")]
    move_uci: String,
}

#[derive(Serialize)]
struct MoveResponse {
    success: bool,
    fen: String,
    legal_moves: Vec<String>,
    is_game_over: bool,
    explanation: Option<String>,
}

#[derive(Deserialize)]
struct EngineMoveRequest {
    depth: Option<u32>,
}

#[derive(Serialize)]
struct EngineMoveResponse {
    #[serde(rename = "move")]
    mv: String,
    explanation: String,
    features: BTreeMap<String, f32>,
}

#[derive(Serialize)]
struct AnalysisResponse {
    features: Vec<(String, String, f32)>, // (raw_name, display_label, value)
    fen: String,
}

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    engine_available: bool,
    version: String,
}

#[derive(Serialize)]
struct EngineStatusResponse {
    model_ready: bool,
    error: bool,
    engine_available: bool,
}

async fn get_dashboard(State(_state): State<SharedState>) -> impl IntoResponse {
    let mut tera = Tera::default();
    let current_dir = std::env::current_dir().unwrap_or_default();

    let template_paths = [
        current_dir.join("web/templates/dashboard.html"),
        current_dir.join("../web/templates/dashboard.html"),
    ];

    let mut loaded_error = String::new();
    let mut loaded = false;
    for path in &template_paths {
        if let Some(path_str) = path.to_str() {
            match tera.add_template_file(path_str, Some("dashboard.html")) {
                Ok(_) => {
                    loaded = true;
                    break;
                }
                Err(e) => {
                    loaded_error.push_str(&format!("{}: {}; ", path_str, e));
                }
            }
        }
    }

    if !loaded {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!(
                "Failed to load dashboard.html template. Errors: {}",
                loaded_error
            ),
        )
            .into_response();
    }

    let context = Context::new();
    match tera.render("dashboard.html", &context) {
        Ok(html) => Html(html).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to render template: {}", e),
        )
            .into_response(),
    }
}

fn get_board_state(s: &GameState) -> BoardState {
    let fen =
        shakmaty::fen::Fen::from_position(&s.board, shakmaty::EnPassantMode::Always).to_string();
    let legal_moves = s
        .board
        .legal_moves()
        .iter()
        .map(|m| {
            shakmaty::uci::UciMove::from_move(*m, shakmaty::CastlingMode::Standard).to_string()
        })
        .collect();

    BoardState {
        fen,
        legal_moves,
        is_game_over: s.board.is_game_over(),
        result: if s.board.is_game_over() {
            Some(format!("{:?}", s.board.outcome()))
        } else {
            None
        },
        turn: if s.board.turn().is_white() {
            "white".to_string()
        } else {
            "black".to_string()
        },
    }
}

async fn get_state_handler(State(state): State<SharedState>) -> Json<BoardState> {
    let s = state.read().unwrap();
    Json(get_board_state(&s))
}

async fn new_game_handler(State(state): State<SharedState>) -> Json<BoardState> {
    {
        let mut s = state.write().unwrap();
        s.reset();
    }
    get_state_handler(State(state)).await
}

async fn make_move_handler(
    State(state): State<SharedState>,
    Json(req): Json<MoveRequest>,
) -> impl IntoResponse {
    let mut s = state.write().unwrap();
    let uci_move: shakmaty::uci::UciMove = match req.move_uci.parse() {
        Ok(m) => m,
        Err(_) => return (StatusCode::BAD_REQUEST, "Invalid UCI").into_response(),
    };

    let m = match uci_move.to_move(&s.board) {
        Ok(m) => m,
        Err(_) => return (StatusCode::BAD_REQUEST, "Illegal move").into_response(),
    };

    if !s.board.legal_moves().contains(&m) {
        return (StatusCode::BAD_REQUEST, "Illegal move").into_response();
    }

    // Sync engine if it exists
    if let Some(engine_arc) = &s.engine {
        let mut engine = engine_arc.write().unwrap();
        let _ = engine.make_move(&uci_move.to_string());
    }

    let explanation = generate_explanation(&s, m);

    let fen_before =
        shakmaty::fen::Fen::from_position(&s.board, shakmaty::EnPassantMode::Always).to_string();
    s.history.push(fen_before);

    s.board.play_unchecked(m);

    let fen =
        shakmaty::fen::Fen::from_position(&s.board, shakmaty::EnPassantMode::Always).to_string();
    let legal_moves = s
        .board
        .legal_moves()
        .iter()
        .map(|m| {
            shakmaty::uci::UciMove::from_move(*m, shakmaty::CastlingMode::Standard).to_string()
        })
        .collect();

    Json(MoveResponse {
        success: true,
        fen,
        legal_moves,
        is_game_over: s.board.is_game_over(),
        explanation,
    })
    .into_response()
}

async fn engine_move_handler(
    State(state): State<SharedState>,
    Json(req): Json<EngineMoveRequest>,
) -> impl IntoResponse {
    let s = state.read().unwrap();
    if s.board.is_game_over() {
        return (StatusCode::BAD_REQUEST, "Game over").into_response();
    }

    let depth = req.depth.unwrap_or(12);
    let engine_arc = match &s.engine {
        Some(e) => e.clone(),
        None => return (StatusCode::SERVICE_UNAVAILABLE, "Engine not available").into_response(),
    };

    let mv_uci = {
        let mut engine = engine_arc.write().unwrap();
        engine.get_best_move(depth).unwrap()
    };

    let uci_move: shakmaty::uci::UciMove = mv_uci.parse().unwrap();
    let m = uci_move.to_move(&s.board).unwrap();
    let explanation =
        generate_explanation(&s, m).unwrap_or_else(|| "No explanation available".to_string());

    let feats = extract_features(&s.board);

    Json(EngineMoveResponse {
        mv: mv_uci,
        explanation,
        features: feats,
    })
    .into_response()
}

async fn undo_move_handler(State(state): State<SharedState>) -> Response {
    let mut s = state.write().unwrap();

    // Undo Turn: Pop twice if possible (Engine move + Player move)
    let mut undone = false;
    for _ in 0..2 {
        if let Some(fen_str) = s.history.pop() {
            let setup: shakmaty::fen::Fen = fen_str.parse().unwrap();
            let board: Chess = setup
                .into_position(shakmaty::CastlingMode::Standard)
                .unwrap();
            s.board = board;

            // Sync engine
            if let Some(engine_arc) = &s.engine {
                let mut engine = engine_arc.write().unwrap();
                let _ = engine.set_position(&fen_str);
            }
            undone = true;
        }
    }

    if undone {
        Json(get_board_state(&s)).into_response()
    } else {
        (StatusCode::BAD_REQUEST, "No moves to undo").into_response()
    }
}

async fn analyze_features_handler(State(state): State<SharedState>) -> Json<AnalysisResponse> {
    let s = state.read().unwrap();
    let feats = extract_features(&s.board);
    let fen =
        shakmaty::fen::Fen::from_position(&s.board, shakmaty::EnPassantMode::Always).to_string();

    let formatted_features = if let Some(model) = &s.model {
        let explainer = SurrogateExplainer::new(model.clone());
        explainer.get_formatted_features(&feats)
    } else {
        feats
            .into_iter()
            .map(|(k, v)| (k.clone(), k, v))
            .collect::<Vec<_>>()
    };

    Json(AnalysisResponse {
        features: formatted_features,
        fen,
    })
}

async fn health_handler(State(state): State<SharedState>) -> Json<HealthResponse> {
    let s = state.read().unwrap();
    Json(HealthResponse {
        status: "healthy".to_string(),
        engine_available: s.engine.is_some(),
        version: "1.0.0".to_string(),
    })
}

async fn engine_status_handler(State(state): State<SharedState>) -> Json<EngineStatusResponse> {
    let s = state.read().unwrap();
    Json(EngineStatusResponse {
        model_ready: s.model_ready,
        error: s.training_error,
        engine_available: s.engine.is_some(),
    })
}

fn generate_explanation(s: &GameState, m: shakmaty::Move) -> Option<String> {
    if let Some(model) = &s.model {
        let explainer = SurrogateExplainer::new(model.clone());
        let mut board_after = s.board.clone();
        board_after.play_unchecked(m);

        let feats_after = extract_features(&board_after);
        let reasons = explainer.explain_move(&feats_after, 3, 0.05);
        if !reasons.is_empty() {
            return Some(
                reasons
                    .iter()
                    .map(|(_, _, text)| format!("- {}", text))
                    .collect::<Vec<_>>()
                    .join("\n"),
            );
        }
    }
    None
}

pub async fn start_server(stockfish_path: String, host: String, port: u16) -> Result<()> {
    let state = Arc::new(RwLock::new(GameState::new(stockfish_path.clone())));

    // Background Initialization
    let state_clone = state.clone();
    tokio::spawn(async move {
        println!("Initializing engine in background...");
        match ExplainableEngine::new(&stockfish_path) {
            Ok(engine) => {
                let engine_arc = Arc::new(RwLock::new(engine));
                {
                    let mut s = state_clone.write().unwrap();
                    s.engine = Some(engine_arc);
                }

                // Try to load existing model
                if std::path::Path::new("model.json").exists() {
                    println!("Loading existing model.json...");
                    if let Ok(model_str) = std::fs::read_to_string("model.json") {
                        if let Ok(model) = serde_json::from_str::<PhaseEnsemble>(&model_str) {
                            let mut s = state_clone.write().unwrap();
                            s.model = Some(model);
                            s.model_ready = true;
                            println!("Model loaded successfully.");
                        }
                    }
                }

                // If no model or failed to load, train one?
                // The Flask app trains one if needed. Let's match that.
                if !state_clone.read().unwrap().model_ready {
                    println!("No model found. Starting background training (100 positions)...");
                    match train_surrogate_model(&stockfish_path, 100) {
                        Ok(ensemble) => {
                            let mut s = state_clone.write().unwrap();
                            s.model = Some(ensemble);
                            s.model_ready = true;
                            println!("Background training complete.");
                            // Save it
                            if let Ok(json) =
                                serde_json::to_string_pretty(&s.model.as_ref().unwrap())
                            {
                                let _ = std::fs::write("model.json", json);
                            }
                        }
                        Err(e) => {
                            let mut s = state_clone.write().unwrap();
                            s.training_error = true;
                            println!("Background training failed: {}", e);
                        }
                    }
                }
            }
            Err(e) => {
                let mut s = state_clone.write().unwrap();
                s.training_error = true;
                println!("Failed to initialize engine: {}", e);
            }
        }
    });

    let current_dir = std::env::current_dir().unwrap_or_default();
    let static_path = if current_dir.join("web/static").exists() {
        current_dir.join("web/static")
    } else {
        current_dir.join("../web/static")
    };

    let app = Router::new()
        .route("/", get(get_dashboard))
        .route("/api/game/state", get(get_state_handler))
        .route("/api/game/new", post(new_game_handler))
        .route("/api/game/move", post(make_move_handler))
        .route("/api/engine/move", post(engine_move_handler))
        .route("/api/analysis/features", get(analyze_features_handler))
        .route("/api/health", get(health_handler))
        .route("/api/engine/status", get(engine_status_handler))
        .route("/api/game/undo", post(undo_move_handler))
        .nest_service("/static", ServeDir::new(static_path))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(format!("{}:{}", host, port)).await?;
    println!("Server running at http://{}:{}", host, port);
    axum::serve(listener, app).await?;

    Ok(())
}
