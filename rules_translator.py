"""
自然语言 → 规则 JSON 翻译引擎 (Rules Translator)

利用本地 Ollama (gemma3:4b) 将用户的自然语言描述
转化为 UniversalGame 解释器可加载的 JSON 配置文件。

流水线:
    用户输入 → System Prompt + Few-shot → LLM → JSON 提取 → 校验 → 保存到 rules/

用法:
    python rules_translator.py
    python rules_translator.py --prompt "5x5 棋盘，落子，四连赢"
    python rules_translator.py --non-interactive --prompt "..."
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, Optional

import requests

# ================================================================
#  常量
# ================================================================

# Ollama 后端
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "gemma3:1b"

LMSTUDIO_BASE_URL = "http://localhost:1234"
LMSTUDIO_MODEL = "gemma-3-4b-instruct"

# OpenAI / OpenAI-compatible 后端，例如 GPT 或 OpenClaw
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

RULES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rules")

# 后端选择: 'ollama', 'lmstudio' 或 'openai'
BACKEND = os.getenv("RULES_LLM_BACKEND", "openai")

# ================================================================
#  System Prompt — 强约束 Few-shot 模板
# ================================================================

SYSTEM_PROMPT = r"""You are a **Board Game Rule Compiler**. Your ONLY job is to convert a natural-language game description into a single JSON object that conforms EXACTLY to the schema below. Output NOTHING except the JSON — no explanations, no markdown fences, no commentary.

## JSON Schema

```
{
  "id":   "<snake_case unique identifier>",
  "name": "<Human-readable name>",
  "board": {
    "geometry": "<square | hexagonal>",
    "size": <int>,
    "canonical_transform": "<negate | flip_vertical | transpose>",
    "adjacency": [[dr,dc], ...]          // ONLY for hexagonal; omit for square
  },
  "pieces": {
    "initial_rows": {                     // omit entire "pieces" block for place-mode games
      "player_1":    [<row>, ...],
      "player_neg1": [<row>, ...]
    }
  },
  "actions": {
    "mode": "<place | move_from_to>",
    "move_vectors": [[dr,dc], ...],       // ONLY for move_from_to; omit for place
    "capture_rules": {                    // ONLY for move_from_to; omit for place
      "straight_capture": <bool>,
      "diagonal_capture_enemy_only": <bool>
    }
  },
  "end_condition": {
    "type": "<reach_rank | n_in_a_row | connectivity>",
    // type-specific fields:
    "target_rank": <int>,   "elimination": <bool>,          // reach_rank
    "n": <int>,                                              // n_in_a_row
    "player_1_axis": "<vertical|horizontal>", "player_neg1_axis": "..." // connectivity
  }
}
```

## Allowed Values

- **geometry**: `"square"` (4/8 neighbours) or `"hexagonal"` (6 neighbours, axial coords)
- **canonical_transform**:
  - `"flip_vertical"` — for move-from-to games where pieces advance toward opposite rank
  - `"transpose"` — for hexagonal connectivity games
  - `"negate"` — default for symmetric place games
- **action mode**: `"place"` (put piece on empty cell) or `"move_from_to"` (slide existing piece)
- **move_vectors**: relative `[dr, dc]` offsets from the piece, e.g. `[-1,-1]` = up-left
- **end_condition type**: `"reach_rank"`, `"n_in_a_row"`, or `"connectivity"`

## Few-shot Examples

### Example 1
User: "8x8 board, two rows of pieces per side, pieces move forward or forward-diagonal, can only capture diagonally, wins by reaching the far side or eliminating all enemy pieces"
Output:
{"id":"breakthrough","name":"Breakthrough","board":{"geometry":"square","size":8,"canonical_transform":"flip_vertical"},"pieces":{"initial_rows":{"player_1":[6,7],"player_neg1":[0,1]}},"actions":{"mode":"move_from_to","move_vectors":[[-1,-1],[-1,0],[-1,1]],"capture_rules":{"straight_capture":false,"diagonal_capture_enemy_only":true}},"end_condition":{"type":"reach_rank","target_rank":7,"elimination":true}}

### Example 2
User: "3x3, place pieces, first to get 3 in a row wins"
Output:
{"id":"tictactoe","name":"TicTacToe","board":{"geometry":"square","size":3,"canonical_transform":"negate"},"pieces":{},"actions":{"mode":"place"},"end_condition":{"type":"n_in_a_row","n":3}}

### Example 3
User: "7x7 hex board, place stones, first to connect top-to-bottom wins"
Output:
{"id":"hex_7x7","name":"Hex","board":{"geometry":"hexagonal","size":7,"canonical_transform":"transpose","adjacency":[[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0]]},"pieces":{},"actions":{"mode":"place"},"end_condition":{"type":"connectivity","player_1_axis":"vertical","player_neg1_axis":"horizontal"}}

## Rules
1. Output ONLY the raw JSON object. No markdown, no ```json fences, no extra text.
2. All field names and enum values must match the schema exactly.
3. If the user description is ambiguous, make reasonable defaults (e.g. square geometry, negate canonical).
4. The "id" must be unique snake_case derived from the game name.
5. For move_from_to games, player_1 always moves in the NEGATIVE row direction (upward).
"""

# ================================================================
#  LLM 后端客户端（支持 Ollama + LM Studio）
# ================================================================

class OllamaError(RuntimeError):
    pass


def check_service(backend: str = BACKEND, base_url: str = None, timeout_s: float = 5.0) -> bool:
    """检查 LLM 服务是否可达"""
    if backend == "lmstudio":
        url = f"{(base_url or LMSTUDIO_BASE_URL).rstrip('/')}/v1/models"
    elif backend == "openai":
        if not OPENAI_API_KEY:
            return False
        url = f"{(base_url or OPENAI_BASE_URL).rstrip('/')}/v1/models"
    else:
        url = f"{(base_url or OLLAMA_BASE_URL).rstrip('/')}/api/tags"

    try:
        headers = {}
        if backend == "openai":
            headers["Authorization"] = f"Bearer {OPENAI_API_KEY}"
        r = requests.get(url, headers=headers, timeout=timeout_s)
        return r.status_code == 200
    except requests.RequestException:
        return False



def chat_once(
    user_prompt: str,
    system_prompt: str = SYSTEM_PROMPT,
    model: str = None,
    base_url: str = None,
    timeout_s: float = 120.0,
    temperature: float = 0.1,
    backend: str = BACKEND,
) -> str:
    """单轮非流式对话，自动根据 backend 选择 API 格式"""
    if backend in ("lmstudio", "openai"):
        # OpenAI 兼容接口
        if backend == "openai":
            _base = (base_url or OPENAI_BASE_URL).rstrip('/')
            _model = model or OPENAI_MODEL
        else:
            _base = (base_url or LMSTUDIO_BASE_URL).rstrip('/')
            _model = model or LMSTUDIO_MODEL

        url = f"{_base}/v1/chat/completions"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        payload = {
            "model": _model,
            "stream": False,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 2048,
        }
        headers = {}
        if backend == "openai":
            if not OPENAI_API_KEY:
                raise OllamaError("OPENAI_API_KEY is not set.")
            headers["Authorization"] = f"Bearer {OPENAI_API_KEY}"

        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
        except requests.RequestException as e:
            raise OllamaError(f"POST /v1/chat/completions failed: {e}") from e
        if r.status_code != 200:
            raise OllamaError(f"/v1/chat/completions returned HTTP {r.status_code}: {r.text}")
        data = r.json()
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content")
    else:
        # Ollama 原生接口
        _base = (base_url or OLLAMA_BASE_URL).rstrip('/')
        _model = model or OLLAMA_MODEL
        url = f"{_base}/api/chat"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        payload = {
            "model": _model,
            "stream": False,
            "messages": messages,
            "options": {"temperature": temperature, "num_ctx": 4096},
        }
        try:
            r = requests.post(url, json=payload, timeout=timeout_s)
        except requests.RequestException as e:
            raise OllamaError(f"POST /api/chat failed: {e}") from e
        if r.status_code != 200:
            raise OllamaError(f"/api/chat returned HTTP {r.status_code}: {r.text}")
        data = r.json()
        content = (data.get("message") or {}).get("content")

    if not isinstance(content, str):
        raise OllamaError(f"Unexpected response shape: {json.dumps(data, indent=2)}")
    return content


# ================================================================
#  JSON 提取与校验
# ================================================================

# 必须存在的顶层字段
REQUIRED_KEYS = {"id", "name", "board", "actions", "end_condition"}
VALID_GEOMETRIES = {"square", "hexagonal", "hex", "hexagonal_axial"}
VALID_MODES = {"place", "move_from_to"}
VALID_END_TYPES = {"reach_rank", "n_in_a_row", "connectivity"}
VALID_TRANSFORMS = {"negate", "flip_vertical", "transpose"}


def extract_json(raw: str) -> Dict[str, Any]:
    """
    从 LLM 输出中提取 JSON 对象。
    处理可能存在的 markdown 代码块标记。
    """
    text = raw.strip()

    # 尝试去除 markdown ```json ... ``` 包裹
    md_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if md_match:
        text = md_match.group(1).strip()

    # 兜底: 找第一个 { ... 最后一个 }
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        text = text[first_brace:last_brace + 1]

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            if len(parsed) == 1 and isinstance(parsed[0], dict):
                parsed = parsed[0]
            else:
                raise ValueError(
                    f"Expected one JSON object, got a list with {len(parsed)} items."
                )
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected JSON object, got {type(parsed).__name__}.")
        if isinstance(parsed.get("pieces"), list) or parsed.get("pieces") is None:
            parsed["pieces"] = {}
        return parsed
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON from LLM output: {e}\nRaw:\n{raw}")


def validate_rule(rule: Dict[str, Any]) -> list[str]:
    """
    校验规则 JSON 是否符合 UniversalGame schema。
    返回错误列表（空列表 = 通过）。
    """
    errors = []

    if not isinstance(rule, dict):
        return [f"Rule must be a JSON object, got {type(rule).__name__}"]

    # 1. 顶层字段检查
    missing = REQUIRED_KEYS - set(rule.keys())
    if missing:
        errors.append(f"Missing required fields: {missing}")

    # 2. board 校验
    board = rule.get("board", {})
    geo = board.get("geometry", "")
    if geo not in VALID_GEOMETRIES:
        errors.append(f"Invalid geometry '{geo}'. Must be one of {VALID_GEOMETRIES}")
    if not isinstance(board.get("size"), int) or board.get("size", 0) < 2:
        errors.append(f"board.size must be an integer >= 2, got {board.get('size')}")
    ct = board.get("canonical_transform", "negate")
    if ct not in VALID_TRANSFORMS:
        errors.append(f"Invalid canonical_transform '{ct}'. Must be one of {VALID_TRANSFORMS}")

    # 3. actions 校验
    actions = rule.get("actions", {})
    mode = actions.get("mode", "")
    if mode not in VALID_MODES:
        errors.append(f"Invalid action mode '{mode}'. Must be one of {VALID_MODES}")
    if mode == "move_from_to":
        vecs = actions.get("move_vectors")
        if not isinstance(vecs, list) or len(vecs) == 0:
            errors.append("move_from_to mode requires non-empty move_vectors list")
        else:
            for v in vecs:
                if not isinstance(v, list) or len(v) != 2:
                    errors.append(f"Each move_vector must be [dr,dc], got {v}")
                    break
        # move_from_to 游戏应使用 flip_vertical
        if ct == "negate":
            errors.append("move_from_to games should use canonical_transform='flip_vertical', not 'negate'")

    # 4. pieces.initial_rows 行号边界校验
    board_size = board.get("size", 0) if isinstance(board.get("size"), int) else 0
    pieces = rule.get("pieces", {})
    init_rows = pieces.get("initial_rows", {})
    for player_key in ("player_1", "player_neg1"):
        rows = init_rows.get(player_key, [])
        for row in rows:
            if not isinstance(row, int) or row < 0 or row >= board_size:
                errors.append(f"pieces.initial_rows.{player_key} row {row} is out of bounds for board size {board_size} (valid: 0..{board_size-1})")

    # 5. end_condition 校验
    ec = rule.get("end_condition", {})
    etype = ec.get("type", "")
    if etype not in VALID_END_TYPES:
        errors.append(f"Invalid end_condition type '{etype}'. Must be one of {VALID_END_TYPES}")
    if etype == "n_in_a_row" and (not isinstance(ec.get("n"), int) or ec["n"] < 2):
        errors.append(f"n_in_a_row requires integer n >= 2, got {ec.get('n')}")
    if etype == "reach_rank" and "target_rank" not in ec:
        errors.append("reach_rank requires target_rank field")

    return errors


# ================================================================
#  翻译流水线 (Translation Pipeline)
# ================================================================

def translate_to_json(
    user_input: str,
    max_retries: int = 2,
    model: str = None,
    base_url: str = None,
    backend: str = BACKEND,
    verbose: bool = True,
) -> Dict[str, Any]:

    """
    主翻译函数: 自然语言 → 结构化 JSON

    包含 Self-Correction 重试机制:
    如果首次输出格式错误，将错误信息反馈给 LLM 让其修正。
    """
    if verbose:
        print(f"[Translator] Sending to {model}...")

    # 第一次尝试
    t0 = time.time()
    raw = chat_once(user_input, model=model, base_url=base_url, backend=backend)
    dt = time.time() - t0
    if verbose:
        print(f"[Translator] Response received ({dt:.1f}s)")

    for attempt in range(max_retries + 1):
        # 提取 JSON
        try:
            rule = extract_json(raw)
        except ValueError as e:
            if attempt >= max_retries:
                raise
            # Self-correction: 反馈解析错误
            correction_prompt = (
                f"Your previous output was not valid JSON. Error: {e}\n"
                f"Please output ONLY the corrected raw JSON object for this game description:\n"
                f"{user_input}"
            )
            if verbose:
                print(f"[Translator] JSON parse failed, retrying ({attempt+1}/{max_retries})...")
            raw = chat_once(correction_prompt, model=model, base_url=base_url, backend=backend)
            continue

        # 校验 schema
        errors = validate_rule(rule)
        if not errors:
            if verbose:
                print(f"[Translator] Validation passed.")
            return rule

        if attempt >= max_retries:
            raise ValueError(
                f"Rule validation failed after {max_retries} retries.\n"
                f"Errors: {errors}\n"
                f"Generated JSON: {json.dumps(rule, indent=2)}"
            )

        # Self-correction: 反馈校验错误
        correction_prompt = (
            f"Your JSON has schema errors: {errors}\n"
            f"The original game description was: {user_input}\n"
            f"Please output ONLY the corrected raw JSON object."
        )
        if verbose:
            print(f"[Translator] Validation errors: {errors}")
            print(f"[Translator] Requesting correction ({attempt+1}/{max_retries})...")
        raw = chat_once(correction_prompt, model=model, base_url=base_url, backend=backend)

    # Should not reach here
    raise RuntimeError("Translation failed")


def save_rule(rule: Dict[str, Any], verbose: bool = True) -> str:
    """将规则 JSON 保存到 rules/ 目录，返回文件路径"""
    os.makedirs(RULES_DIR, exist_ok=True)
    game_id = rule.get("id", "unnamed_game")
    filename = f"{game_id}.json"
    filepath = os.path.join(RULES_DIR, filename)

    # 防止覆盖已有文件（除非用户确认）
    if os.path.exists(filepath):
        if verbose:
            print(f"[Translator] Warning: {filepath} already exists, adding suffix")
        base, ext = os.path.splitext(filepath)
        counter = 2
        while os.path.exists(f"{base}_{counter}{ext}"):
            counter += 1
        filepath = f"{base}_{counter}{ext}"

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(rule, f, ensure_ascii=False, indent=2)
        f.write("\n")

    if verbose:
        print(f"[Translator] Saved: {filepath}")
    return filepath


# ================================================================
#  集成验证 (Integration Check)
# ================================================================

def verify_integration(json_path: str, verbose: bool = True) -> bool:
    """尝试用 UniversalGame 加载生成的 JSON，验证是否能成功初始化"""
    project_dir = os.path.dirname(os.path.abspath(__file__))
    games_dir = os.path.join(project_dir, "games")
    if games_dir not in sys.path:
        sys.path.insert(0, games_dir)
    from universal_game import UniversalGame

    rel_path = os.path.relpath(json_path, games_dir)
    try:
        game = UniversalGame(rel_path)
        board_size = game.get_board_size()
        action_size = game.get_action_size()
        if verbose:
            print(f"[Integration] Game initialized: {game.name}")
            print(f"[Integration]   Board: {board_size[0]}x{board_size[1]}")
            print(f"[Integration]   Action size: {action_size}")
            print(f"[Integration]   End condition: {game.end_condition.get('type')}")

            # 快速功能检查
            board = game.get_initial_board()
            valids = game.get_valid_moves(board, 1)
            n_valid = int(valids.sum())
            print(f"[Integration]   Initial valid moves: {n_valid}")
        return True
    except Exception as e:
        if verbose:
            print(f"[Integration] FAILED: {e}")
        return False


# ================================================================
#  CLI 主入口
# ================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="Natural Language -> Game Rule JSON Translator")
    ap.add_argument("--prompt", type=str, default=None,
                    help="Game description (skip interactive mode)")
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--non-interactive", action="store_true",
                    help="No interactive prompts; requires --prompt")
    ap.add_argument("--skip-verify", action="store_true",
                    help="Skip UniversalGame integration check")
    args = ap.parse_args()

    # Step 0: 检查 Ollama 服务
    print("=" * 60)
    print("  Rules Translator — NL to JSON Pipeline")
    print("=" * 60)
    print(f"\n[Step 0] Checking LLM service backend={BACKEND}...")
    if not check_service():
        print("ERROR: LLM service not reachable. Check API key, base URL, or local server.")
        return 1
    print("  OK — Ollama is online.")

    # Step 1: 获取用户输入
    if args.prompt:
        user_input = args.prompt
    elif args.non_interactive:
        print("ERROR: --non-interactive requires --prompt")
        return 1
    else:
        print("\n[Step 1] Describe your game in natural language.")
        print("  Examples:")
        print('    "5x5 board, place pieces, 4 in a row wins"')
        print('    "6x6, two rows of pieces, move forward/diagonal, reach other side to win"')
        print()
        user_input = input("  Your description: ").strip()
        if not user_input:
            print("No input provided. Exiting.")
            return 1

    # Step 2: 翻译
    print(f"\n[Step 2] Translating with {args.model or OPENAI_MODEL if BACKEND == 'openai' else args.model or OLLAMA_MODEL}...")
    try:
        rule = translate_to_json(user_input, model=args.model)
    except (OllamaError, ValueError) as e:
        print(f"\nERROR: Translation failed.\n{e}")
        return 1

    print(f"\n--- Generated JSON ---")
    print(json.dumps(rule, ensure_ascii=False, indent=2))

    # Step 3: 保存
    print(f"\n[Step 3] Saving to rules/...")
    filepath = save_rule(rule)

    # Step 4: 集成验证
    if not args.skip_verify:
        print(f"\n[Step 4] Integration verification...")
        ok = verify_integration(filepath)
        if ok:
            print("\n  SUCCESS — Game is ready for UniversalNet training!")
        else:
            print("\n  FAILED — Generated JSON could not initialize UniversalGame.")
            return 1
    else:
        print("\n[Step 4] Skipped (--skip-verify)")

    print("\n" + "=" * 60)
    print(f"  Pipeline complete. File: {filepath}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)
