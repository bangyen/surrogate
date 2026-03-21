use shakmaty::{attacks, Bitboard, Chess, Color, Position, Role, Square};
use std::collections::BTreeMap;
use crate::pawn_cache::{pawn_cache, pawn_zobrist, PawnCacheEntry, PAWN_CACHE_SIZE};

pub fn extract(pos: &Chess, feats: &mut BTreeMap<String, f32>, turn: Color, opp: Color) {
    let board = pos.board();

    // Helper: is_passed
    let is_passed = |sq: Square, side: Color| {
        let file = sq.file();
        let rank = sq.rank();
        let enemy_pawns = board.by_role(Role::Pawn) & board.by_color(side.other());

        for f_off in -1..=1 {
            let f = file as i8 + f_off;
            if !(0..=7).contains(&f) {
                continue;
            }
            let check_file = shakmaty::File::new(f as u32);
            let file_bb = Bitboard::from_file(check_file);
            let ahead_bb = match side {
                Color::White => {
                    let mut bb = Bitboard::EMPTY;
                    for r in (rank as usize + 1)..8 {
                        bb |= Bitboard::from_rank(shakmaty::Rank::new(r as u32));
                    }
                    bb
                }
                Color::Black => {
                    let mut bb = Bitboard::EMPTY;
                    for r in 0..(rank as usize) {
                        bb |= Bitboard::from_rank(shakmaty::Rank::new(r as u32));
                    }
                    bb
                }
            };
            if !(enemy_pawns & file_bb & ahead_bb).is_empty() {
                return false;
            }
        }
        true
    };

    // Pawn structure features (cached by pawn Zobrist hash)
    let pawn_hash = pawn_zobrist(board);
    let pawn_key = pawn_hash ^ if turn == Color::White { 0 } else { 0xAAAA_AAAA_AAAA_AAAA };

    let pawn_feats = {
        let cache = pawn_cache().lock().unwrap();
        let idx = (pawn_key as usize) % PAWN_CACHE_SIZE;
        if cache[idx].key == pawn_key {
            Some(cache[idx].clone())
        } else {
            None
        }
    };

    let pawn_feats = pawn_feats.unwrap_or_else(|| {
        let count_isolated_fn = |side: Color| -> f32 {
            let mut count = 0;
            let my_pawns = board.by_role(Role::Pawn) & board.by_color(side);
            for sq in my_pawns {
                let file: shakmaty::File = sq.file();
                let mut has_neighbor = false;
                for f_off in [-1, 1] {
                    let f = file as i32 + f_off;
                    if (0..8).contains(&f) {
                        let adj_file = shakmaty::File::new(f as u32);
                        let adj_bb = Bitboard::from_file(adj_file);
                        if !(board.by_role(Role::Pawn) & board.by_color(side) & adj_bb).is_empty() {
                            has_neighbor = true;
                            break;
                        }
                    }
                }
                if !has_neighbor {
                    count += 1;
                }
            }
            count as f32
        };

        let count_doubled_fn = |side: Color| -> f32 {
            let my_pawns = board.by_role(Role::Pawn) & board.by_color(side);
            let mut count = 0.0;
            for f in 0..8 {
                let file_bb = Bitboard::from_file(shakmaty::File::new(f as u32));
                let pawns_on_file = (my_pawns & file_bb).count();
                if pawns_on_file >= 2 {
                    count += (pawns_on_file - 1) as f32;
                }
            }
            count
        };

        let count_backward_fn = |side: Color| -> f32 {
            let mut count = 0;
            let my_pawns = board.by_role(Role::Pawn) & board.by_color(side);
            let opp_side = side.other();
            let mut enemy_pawn_attacks = Bitboard::EMPTY;
            for sq in board.by_role(Role::Pawn) & board.by_color(opp_side) {
                enemy_pawn_attacks |= attacks::pawn_attacks(opp_side, sq);
            }
            for sq in my_pawns {
                let file: shakmaty::File = sq.file();
                let rank: shakmaty::Rank = sq.rank();
                let rank_usize = rank as usize;
                let mut is_supported = false;
                for f_off in [-1, 1] {
                    let f = file as i32 + f_off;
                    if (0..8).contains(&f) {
                        let adj_file = shakmaty::File::new(f as u32);
                        let adj_bb = Bitboard::from_file(adj_file);
                        let adj_pawns = board.by_role(Role::Pawn) & board.by_color(side) & adj_bb;
                        for p_sq in adj_pawns {
                            let p_rank = p_sq.rank();
                            if (side == Color::White && (p_rank as usize) <= rank_usize)
                                || (side == Color::Black && (p_rank as usize) >= rank_usize)
                            {
                                is_supported = true;
                                break;
                            }
                        }
                    }
                    if is_supported {
                        break;
                    }
                }
                if is_supported {
                    continue;
                }
                let stop_rank = if side == Color::White {
                    sq.rank().offset(1)
                } else {
                    sq.rank().offset(-1)
                };
                if let Some(r) = stop_rank {
                    let stop_sq = Square::from_coords(file, r);
                    if !(enemy_pawn_attacks & Bitboard::from_square(stop_sq)).is_empty() {
                        count += 1;
                    }
                }
            }
            count as f32
        };

        let count_passed_fn = |side: Color| -> f32 {
            let mut count = 0;
            let my_pawns = board.by_role(Role::Pawn) & board.by_color(side);
            for sq in my_pawns {
                if is_passed(sq, side) {
                    count += 1;
                }
            }
            count as f32
        };

        let pawn_chain_fn = |side: Color| -> f32 {
            let my_pawns = board.by_role(Role::Pawn) & board.by_color(side);
            let mut count = 0.0;
            for sq in my_pawns {
                let pawn_attackers = attacks::pawn_attacks(side.other(), sq) & my_pawns;
                if !pawn_attackers.is_empty() {
                    count += 1.0;
                }
            }
            count
        };

        let entry = PawnCacheEntry {
            key: pawn_key,
            isolated_us: count_isolated_fn(turn),
            isolated_them: count_isolated_fn(opp),
            doubled_us: count_doubled_fn(turn),
            doubled_them: count_doubled_fn(opp),
            backward_us: count_backward_fn(turn),
            backward_them: count_backward_fn(opp),
            passed_us: count_passed_fn(turn),
            passed_them: count_passed_fn(opp),
            pawn_chain_us: pawn_chain_fn(turn),
            pawn_chain_them: pawn_chain_fn(opp),
        };

        if let Ok(mut cache) = pawn_cache().lock() {
            let idx = (pawn_key as usize) % PAWN_CACHE_SIZE;
            cache[idx] = entry.clone();
        }
        entry
    });

    feats.insert("isolated_pawns_us".to_string(), pawn_feats.isolated_us);
    feats.insert("isolated_pawns_them".to_string(), pawn_feats.isolated_them);
    feats.insert("backward_pawns_us".to_string(), pawn_feats.backward_us);
    feats.insert("backward_pawns_them".to_string(), pawn_feats.backward_them);
    feats.insert("passed_us".to_string(), pawn_feats.passed_us);
    feats.insert("passed_them".to_string(), pawn_feats.passed_them);
    feats.insert("doubled_pawns_us".to_string(), pawn_feats.doubled_us);
    feats.insert("doubled_pawns_them".to_string(), pawn_feats.doubled_them);
    feats.insert("pawn_chain_us".to_string(), pawn_feats.pawn_chain_us);
    feats.insert("pawn_chain_them".to_string(), pawn_feats.pawn_chain_them);
}
