/* tslint:disable */
/* eslint-disable */

/**
 * A game the browser can drive.
 */
export class WasmGame {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Static evaluation from the side to move's perspective, in
     * centipawns.
     */
    evaluate(): number;
    /**
     * Explain a move without playing it, as newline-separated reasons.
     *
     * Returns an empty string when no model is loaded or the surrogate
     * has nothing above the noise floor to say.
     */
    explain(uci: string): string;
    /**
     * Extracted features for the current position, as JSON.
     */
    featuresJson(): string;
    /**
     * The current position in FEN notation.
     */
    fen(): string;
    /**
     * Whether explanations are available.
     */
    hasModel(): boolean;
    /**
     * Whether the game has ended under this variant's rules.
     */
    isGameOver(): boolean;
    /**
     * Legal moves in UCI notation.
     */
    legalMoves(): string[];
    /**
     * Load the surrogate model, enabling explanations.
     *
     * Without it the engine still plays; moves simply come back without
     * reasons attached.
     */
    loadModel(model_json: string): void;
    /**
     * Start a new game.  `variant` accepts the same names as the CLI
     * ("standard", "koth", "3check", "antichess").
     */
    constructor(variant: string);
    /**
     * Search for the engine's reply and play it in one step.
     */
    playEngineMove(depth: number): string | undefined;
    /**
     * Play a move given in UCI notation.
     */
    playMove(uci: string): void;
    /**
     * Search for the engine's best move *without* playing it, returning
     * UCI notation.  Returns `None` when there is no legal move.
     *
     * Searching and playing are separate so a caller can explain the
     * move first: `explain` describes a move from the position it is
     * played in, which is gone once the move has been applied.
     *
     * This runs on the caller's thread, so keep `depth` modest -- the
     * browser is unresponsive until it returns.
     */
    searchMove(depth: number): string | undefined;
    /**
     * "white" or "black".
     */
    turn(): string;
}

/**
 * Variants the browser build can offer, as `slug\tdescription` lines.
 */
export function supportedVariants(): string[];

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_wasmgame_free: (a: number, b: number) => void;
    readonly supportedVariants: () => [number, number];
    readonly wasmgame_evaluate: (a: number) => number;
    readonly wasmgame_explain: (a: number, b: number, c: number) => [number, number, number, number];
    readonly wasmgame_featuresJson: (a: number) => [number, number, number, number];
    readonly wasmgame_fen: (a: number) => [number, number];
    readonly wasmgame_hasModel: (a: number) => number;
    readonly wasmgame_isGameOver: (a: number) => number;
    readonly wasmgame_legalMoves: (a: number) => [number, number];
    readonly wasmgame_loadModel: (a: number, b: number, c: number) => [number, number];
    readonly wasmgame_new: (a: number, b: number) => [number, number, number];
    readonly wasmgame_playEngineMove: (a: number, b: number) => [number, number, number, number];
    readonly wasmgame_playMove: (a: number, b: number, c: number) => [number, number];
    readonly wasmgame_searchMove: (a: number, b: number) => [number, number];
    readonly wasmgame_turn: (a: number) => [number, number];
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __externref_drop_slice: (a: number, b: number) => void;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
