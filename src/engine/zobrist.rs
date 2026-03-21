use shakmaty::{Chess, Color, Position, Role, Square};
use std::sync::OnceLock;

/// Maps a (color, role) pair to a 0..11 index for Zobrist table lookup.
/// White pieces occupy indices 0..5, Black pieces 6..11.
#[inline]
pub fn piece_index(color: Color, role: Role) -> usize {
    let c = if color == Color::White { 0 } else { 6 };
    let r = match role {
        Role::Pawn => 0,
        Role::Knight => 1,
        Role::Bishop => 2,
        Role::Rook => 3,
        Role::Queen => 4,
        Role::King => 5,
    };
    c + r
}

/// Pre-computed random keys for Zobrist hashing, generated
/// deterministically from a fixed seed so values are stable across runs.
pub struct ZobristKeys {
    pub pieces: [[u64; 12]; 64],
    pub side_to_move: u64,
    pub castling_sq: [u64; 64],
    pub en_passant: [u64; 8],
}

/// Simple xorshift64 PRNG for deterministic Zobrist key generation.
/// The state must never be zero.
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Lazily initialised global Zobrist key table so that every call to
/// `zobrist_hash` uses the same random constants.
pub fn zobrist_keys() -> &'static ZobristKeys {
    static KEYS: OnceLock<ZobristKeys> = OnceLock::new();
    KEYS.get_or_init(|| {
        let mut rng: u64 = 0x123456789ABCDEF0;
        let mut pieces = [[0u64; 12]; 64];
        for sq in pieces.iter_mut() {
            for pc in sq.iter_mut() {
                *pc = xorshift64(&mut rng);
            }
        }
        let side_to_move = xorshift64(&mut rng);
        let mut castling_sq = [0u64; 64];
        for v in castling_sq.iter_mut() {
            *v = xorshift64(&mut rng);
        }
        let mut en_passant = [0u64; 8];
        for v in en_passant.iter_mut() {
            *v = xorshift64(&mut rng);
        }
        ZobristKeys {
            pieces,
            side_to_move,
            castling_sq,
            en_passant,
        }
    })
}

/// Compute a full Zobrist hash for a chess position, incorporating
/// piece placement, side to move, and castling rights.
pub fn zobrist_hash(pos: &Chess) -> u64 {
    let keys = zobrist_keys();
    let board = pos.board();
    let mut hash = 0u64;

    for sq in Square::ALL {
        if let Some(piece) = board.piece_at(sq) {
            let pi = piece_index(piece.color, piece.role);
            hash ^= keys.pieces[sq as usize][pi];
        }
    }
    if pos.turn() == Color::Black {
        hash ^= keys.side_to_move;
    }
    for sq in pos.castles().castling_rights() {
        hash ^= keys.castling_sq[sq as usize];
    }
    if let Some(ep) = pos.maybe_ep_square() {
        hash ^= keys.en_passant[ep.file() as usize];
    }
    hash
}
