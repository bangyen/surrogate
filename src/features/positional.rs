use shakmaty::{attacks, Bitboard, Chess, Color, Position, Role, Square};
use std::collections::BTreeMap;

pub fn extract(pos: &Chess, feats: &mut BTreeMap<String, f32>, turn: Color, opp: Color) {
    let board = pos.board();
    let phase = feats.get("phase").cloned().unwrap_or(24.0);

    // File State
    let get_file_state = |side: Color| {
        let mut open = 0;
        let mut semi_open = 0;
        for f in 0..8 {
            let file_bb = Bitboard::from_file(shakmaty::File::new(f as u32));
            let pawns_on_file = board.by_role(Role::Pawn) & file_bb;
            let my_pawns = pawns_on_file & board.by_color(side);
            let opp_pawns = pawns_on_file & board.by_color(side.other());

            if pawns_on_file.is_empty() {
                open += 1;
            } else if my_pawns.is_empty() || opp_pawns.is_empty() {
                semi_open += 1;
            }
        }
        (open as f32, semi_open as f32)
    };

    let (of_us, sof_us) = get_file_state(turn);
    let (of_them, sof_them) = get_file_state(opp);
    feats.insert("open_files_us".to_string(), of_us);
    feats.insert("semi_open_us".to_string(), sof_us);
    feats.insert("open_files_them".to_string(), of_them);
    feats.insert("semi_open_them".to_string(), sof_them);

    // Center Control & Piece Activity
    let center_squares = Bitboard::from_square(Square::D4)
        | Bitboard::from_square(Square::D5)
        | Bitboard::from_square(Square::E4)
        | Bitboard::from_square(Square::E5);

    let center_us = (board.occupied() & board.by_color(turn) & center_squares).count() as f32;
    let center_them = (board.occupied() & board.by_color(opp) & center_squares).count() as f32;
    feats.insert("center_control_us".to_string(), center_us);
    feats.insert("center_control_them".to_string(), center_them);

    let calc_activity = |side: Color| {
        let mut attacks = Bitboard::EMPTY;
        for sq in board.by_color(side) {
            attacks |= board.attacks_from(sq);
        }
        attacks.count() as f32
    };
    feats.insert("piece_activity_us".to_string(), calc_activity(turn));
    feats.insert("piece_activity_them".to_string(), calc_activity(opp));

    // Space
    let count_space = |side: Color| {
        let mut controlled = Bitboard::EMPTY;
        for sq in board.by_color(side) {
            controlled |= board.attacks_from(sq);
        }
        let opp_half = if side == Color::White {
            Bitboard::from_rank(shakmaty::Rank::Fifth)
                | Bitboard::from_rank(shakmaty::Rank::Sixth)
                | Bitboard::from_rank(shakmaty::Rank::Seventh)
                | Bitboard::from_rank(shakmaty::Rank::Eighth)
        } else {
            Bitboard::from_rank(shakmaty::Rank::First)
                | Bitboard::from_rank(shakmaty::Rank::Second)
                | Bitboard::from_rank(shakmaty::Rank::Third)
                | Bitboard::from_rank(shakmaty::Rank::Fourth)
        };
        (controlled & opp_half).count() as f32
    };
    feats.insert("space_us".to_string(), count_space(turn));
    feats.insert("space_them".to_string(), count_space(opp));

    // King Tropism
    let king_tropism = |side: Color| {
        let them = side.other();
        let enemy_king = board.king_of(them);
        if enemy_king.is_none() {
            return 0.0;
        }
        let ksq = enemy_king.unwrap();
        let mut tropism = 0.0;
        for sq in board.by_color(side) {
            if let Some(piece) = board.piece_at(sq) {
                if piece.role == Role::King || piece.role == Role::Pawn {
                    continue;
                }
                let dist = ksq.distance(sq) as f32;
                tropism += 7.0 - dist;
            }
        }
        tropism
    };
    feats.insert("king_tropism_us".to_string(), king_tropism(turn));
    feats.insert("king_tropism_them".to_string(), king_tropism(opp));

    // Batteries
    let count_batteries = |side: Color| {
        let mut count = 0;
        for i in 0..8 {
            let mut file_pieces = 0;
            for r in 0..8 {
                let sq = Square::from_coords(
                    shakmaty::File::new(i as u32),
                    shakmaty::Rank::new(r as u32),
                );
                if let Some(p) = board.piece_at(sq) {
                    if p.color == side && (p.role == Role::Rook || p.role == Role::Queen) {
                        file_pieces += 1;
                    }
                }
            }
            if file_pieces >= 2 {
                count += 1;
            }
            let mut rank_pieces = 0;
            for f in 0..8 {
                let sq = Square::from_coords(
                    shakmaty::File::new(f as u32),
                    shakmaty::Rank::new(i as u32),
                );
                if let Some(p) = board.piece_at(sq) {
                    if p.color == side && (p.role == Role::Rook || p.role == Role::Queen) {
                        rank_pieces += 1;
                    }
                }
            }
            if rank_pieces >= 2 {
                count += 1;
            }
        }
        for s in 0..15 {
            let mut diag_pieces = 0;
            for f in 0..8 {
                let r = s - f;
                if (0..8).contains(&r) {
                    let sq = Square::from_coords(
                        shakmaty::File::new(f as u32),
                        shakmaty::Rank::new(r as u32),
                    );
                    if let Some(p) = board.piece_at(sq) {
                        if p.color == side && (p.role == Role::Bishop || p.role == Role::Queen) {
                            diag_pieces += 1;
                        }
                    }
                }
            }
            if diag_pieces >= 2 {
                count += 1;
            }
        }
        for d in -7..8 {
            let mut diag_pieces = 0;
            for f in 0..8 {
                let r = f - d;
                if (0..8).contains(&r) {
                    let sq = Square::from_coords(
                        shakmaty::File::new(f as u32),
                        shakmaty::Rank::new(r as u32),
                    );
                    if let Some(p) = board.piece_at(sq) {
                        if p.color == side && (p.role == Role::Bishop || p.role == Role::Queen) {
                            diag_pieces += 1;
                        }
                    }
                }
            }
            if diag_pieces >= 2 {
                count += 1;
            }
        }
        count as f32
    };
    feats.insert("batteries_us".to_string(), count_batteries(turn));
    feats.insert("batteries_them".to_string(), count_batteries(opp));

    // Rook on Open File
    let rook_on_open = |side: Color| {
        let mut count = 0.0;
        let rooks = board.by_role(Role::Rook) & board.by_color(side);
        for sq in rooks {
            let sq_file: shakmaty::File = sq.file();
            let file_bb = Bitboard::from_file(sq_file);
            let pawns_on_file = board.by_role(Role::Pawn) & file_bb;
            if pawns_on_file.is_empty() {
                count += 1.0;
            } else if (pawns_on_file & board.by_color(side)).is_empty() {
                count += 0.5;
            }
        }
        count
    };
    feats.insert("rook_open_file_us".to_string(), rook_on_open(turn));
    feats.insert("rook_open_file_them".to_string(), rook_on_open(opp));

    // Connected Rooks
    let connected_rooks = |side: Color| -> f32 {
        let rooks: Vec<Square> = (board.by_role(Role::Rook) & board.by_color(side))
            .into_iter()
            .collect();
        if rooks.len() < 2 {
            return 0.0;
        }
        let (r0, r1) = (rooks[0], rooks[1]);
        if r0.rank() != r1.rank() {
            return 0.0;
        }
        let between = attacks::between(r0, r1) & board.occupied();
        if between.is_empty() {
            1.0
        } else {
            0.0
        }
    };
    feats.insert("connected_rooks_us".to_string(), connected_rooks(turn));
    feats.insert("connected_rooks_them".to_string(), connected_rooks(opp));

    // PSTs
    const PST_PAWN: [i16; 64] = [
        0, 0, 0, 0, 0, 0, 0, 0, 50, 50, 50, 50, 50, 50, 50, 50, 10, 10, 20, 30, 30, 20, 10, 10, 5,
        5, 10, 25, 25, 10, 5, 5, 0, 0, 0, 20, 20, 0, 0, 0, 5, -5, -10, 0, 0, -10, -5, 5, 5, 10, 10,
        -20, -20, 10, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0,
    ];
    const PST_KNIGHT: [i16; 64] = [
        -50, -40, -30, -30, -30, -30, -40, -50, -40, -20, 0, 0, 0, 0, -20, -40, -30, 0, 10, 15, 15,
        10, 0, -30, -30, 5, 15, 20, 20, 15, 5, -30, -30, 0, 15, 20, 20, 15, 0, -30, -30, 5, 10, 15,
        15, 10, 5, -30, -40, -20, 0, 5, 5, 0, -20, -40, -50, -40, -30, -30, -30, -30, -40, -50,
    ];
    const PST_BISHOP: [i16; 64] = [
        -20, -10, -10, -10, -10, -10, -10, -20, -10, 0, 0, 0, 0, 0, 0, -10, -10, 0, 5, 10, 10, 5,
        0, -10, -10, 5, 5, 10, 10, 5, 5, -10, -10, 0, 10, 10, 10, 10, 0, -10, -10, 10, 10, 10, 10,
        10, 10, -10, -10, 5, 0, 0, 0, 0, 5, -10, -20, -10, -10, -10, -10, -10, -10, -20,
    ];
    const PST_ROOK: [i16; 64] = [
        0, 0, 0, 0, 0, 0, 0, 0, 5, 10, 10, 10, 10, 10, 10, 5, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0,
        0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0,
        -5, 0, 0, 0, 5, 5, 0, 0, 0,
    ];
    const PST_QUEEN: [i16; 64] = [
        -20, -10, -10, -5, -5, -10, -10, -20, -10, 0, 0, 0, 0, 0, 0, -10, -10, 0, 5, 5, 5, 5, 0,
        -10, -5, 0, 5, 5, 5, 5, 0, -5, -5, 0, 5, 5, 5, 5, 0, -5, -10, 5, 5, 5, 5, 5, 0, -10, -10,
        0, 5, 0, 0, 0, 0, -10, -20, -10, -10, -5, -5, -10, -10, -20,
    ];
    const PST_KING_MG: [i16; 64] = [
        -30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30, -30, -40,
        -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30, -20, -30, -30, -40,
        -40, -30, -30, -20, -10, -20, -20, -20, -20, -20, -20, -10, 20, 20, 0, 0, 0, 0, 20, 20, 20,
        30, 10, 0, 0, 10, 30, 20,
    ];
    const PST_KING_EG: [i16; 64] = [
        -50, -40, -30, -20, -20, -30, -40, -50, -30, -20, -10, 0, 0, -10, -20, -30, -30, -10, 20,
        30, 30, 20, -10, -30, -30, -10, 30, 40, 40, 30, -10, -30, -30, -10, 30, 40, 40, 30, -10,
        -30, -30, -10, 20, 30, 30, 20, -10, -30, -30, -30, 0, 0, 0, 0, -30, -30, -50, -30, -30,
        -30, -30, -30, -30, -50,
    ];

    let mg_phase = (phase.min(24.0)) / 24.0;
    let eg_phase = 1.0 - mg_phase;

    let get_pst_val = |sq: Square, role: Role, side: Color| {
        let idx = if side == Color::White {
            sq.flip_vertical() as usize
        } else {
            sq as usize
        };
        let mg = match role {
            Role::Pawn => PST_PAWN[idx],
            Role::Knight => PST_KNIGHT[idx],
            Role::Bishop => PST_BISHOP[idx],
            Role::Rook => PST_ROOK[idx],
            Role::Queen => PST_QUEEN[idx],
            Role::King => PST_KING_MG[idx],
        };
        let eg = match role {
            Role::Pawn => PST_PAWN[idx],
            Role::Knight => PST_KNIGHT[idx],
            Role::Bishop => PST_BISHOP[idx],
            Role::Rook => PST_ROOK[idx],
            Role::Queen => PST_QUEEN[idx],
            Role::King => PST_KING_EG[idx],
        };
        (mg as f32 * mg_phase + eg as f32 * eg_phase) / 100.0
    };

    let mut pst_us = 0.0;
    let mut pst_them = 0.0;
    for sq in Square::ALL {
        if let Some(piece) = board.piece_at(sq) {
            let val = get_pst_val(sq, piece.role, piece.color);
            if piece.color == turn {
                pst_us += val;
            } else {
                pst_them += val;
            }
        }
    }
    feats.insert("pst_us".to_string(), pst_us);
    feats.insert("pst_them".to_string(), pst_them);
}
