"""Replay packed moves into signed int8 boards."""
from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Tuple

import numpy as np

from .constants import MAX_PLIES

def initial_board() -> np.ndarray:
    board = np.zeros((8, 8), dtype=np.int8)
    board[0] = [4, 2, 3, 5, 6, 3, 2, 4]
    board[1] = 1
    board[6] = -1
    board[7] = [-4, -2, -3, -5, -6, -3, -2, -4]
    return board


def movement_value(value) -> int:
    """Decode uint16 movement, including Arrow/bytes values from Arrow."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return int.from_bytes(bytes(value), "little")
    return int(value)


def board_sequence(moves: Iterable) -> Tuple[np.ndarray, np.ndarray]:
    """Replay standard-chess moves from the initial position.

    Assumes legal source games, matching the PGN writer; this is not a full
    legality validator. Promotion codes follow encodeMove in the Go writer:
    1=N, 2=B, 3=R, 4=Q (internal board IDs are 2, 3, 4, 5).
    """
    board = initial_board()
    # Preserve signed piece IDs; expand to one-hot inside the model on device.
    states = np.zeros((MAX_PLIES, 8, 8), dtype=np.int8)
    valid = np.zeros((MAX_PLIES,), dtype=bool)
    for ply, raw in enumerate(moves if moves is not None else ()):
        if ply >= MAX_PLIES:
            break
        code = movement_value(raw["movement"] if isinstance(raw, dict) else raw)
        src, dst = code & 63, (code >> 6) & 63
        promotion = (code >> 12) & 7
        if promotion > 4:
            raise ValueError(f"잘못된 promotion 코드: {promotion} (ply={ply+1})")
        sy, sx, dy, dx = src // 8, src % 8, dst // 8, dst % 8
        piece = int(board[sy, sx])
        if piece == 0:  # malformed row: keep the previous state and stop safely
            break
        board[sy, sx] = 0
        # A legal pawn diagonal move to an empty square is en passant.
        if abs(piece) == 1 and sx != dx and board[dy, dx] == 0:
            board[sy, dx] = 0
        # Chess castling: move the rook along with the king.
        if abs(piece) == 6 and abs(dx - sx) == 2:
            rook_x = 7 if dx > sx else 0
            board[dy, 5 if dx > sx else 3] = board[dy, rook_x]
            board[dy, rook_x] = 0
        promoted_piece = promotion + 1 if promotion else abs(piece)
        board[dy, dx] = (1 if piece > 0 else -1) * promoted_piece
        states[ply] = board
        valid[ply] = True
    return states, valid


def _decode_into(moves, offsets, present, boards, valid):
    """Numba kernel: sequential games/moves, writes directly into a batch."""
    back_rank = (4, 2, 3, 5, 6, 3, 2, 4)
    for game in range(len(present)):
        boards[game, :, :, :] = 0
        valid[game, :] = False
        if not present[game]:
            continue
        board = np.zeros((8, 8), dtype=np.int8)
        for x in range(8):
            board[0, x] = back_rank[x]
            board[1, x] = 1
            board[6, x] = -1
            board[7, x] = -back_rank[x]
        for ply in range(min(offsets[game+1] - offsets[game], MAX_PLIES)):
            code = int(moves[offsets[game] + ply])
            src, dst = code & 63, (code >> 6) & 63
            promotion = (code >> 12) & 7
            if promotion > 4:
                raise ValueError("Invalid promotion code (expected 0..4)")
            sy, sx, dy, dx = src // 8, src % 8, dst // 8, dst % 8
            piece = int(board[sy, sx])
            if piece == 0:
                break
            board[sy, sx] = 0
            if abs(piece) == 1 and sx != dx and board[dy, dx] == 0:
                board[sy, dx] = 0
            if abs(piece) == 6 and abs(dx - sx) == 2:
                rook_x = 7 if dx > sx else 0
                board[dy, 5 if dx > sx else 3] = board[dy, rook_x]
                board[dy, rook_x] = 0
            promoted = promotion + 1 if promotion else abs(piece)
            board[dy, dx] = (1 if piece > 0 else -1) * promoted
            boards[game, ply, :, :] = board
            valid[game, ply] = True


@lru_cache(maxsize=1)
def compiled_decoder():
    try:
        from numba import njit
    except ImportError as exc:
        raise RuntimeError(
            "Numba가 필요합니다. 프로젝트 의존성을 설치하거나 "
            'decoder="python"으로 실행하세요.') from exc
    # No worker threads, no disk cache, no fastmath, independent of the GPU runtime.
    return njit(nogil=True)(_decode_into)


def readonly_array(values, dtype):
    result = np.ascontiguousarray(values, dtype=dtype).view()
    result.flags.writeable = False
    return result


def warmup_decoder(decoder):
    if decoder == "numba":
        print("[decoder] Numba 순차 복원 컴파일 중 (최초 1회)...", flush=True)
        compiled_decoder()(
            readonly_array([], np.uint16), readonly_array([0, 0], np.int64),
            readonly_array([True], np.bool_),
            np.zeros((1, MAX_PLIES, 8, 8), np.int8), np.zeros((1, MAX_PLIES), bool))
        print("[decoder] 준비 완료", flush=True)
