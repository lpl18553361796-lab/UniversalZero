"""
UniversalZero — Streamlit Interactive Frontend v3

改进:
    - 保留原版 Plotly 棋盘视觉效果
    - 可直接点击棋盘落子 (Plotly on_select)
    - @st.fragment 减少闪烁 (局部刷新，不刷全页)
    - 训练面板: 支持网页端训练模型

运行:
    streamlit run app.py
"""

import os
import sys
import math
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# --- 确保项目路径 ---
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from game import GAME_REGISTRY, get_game_by_id, register_game
from universal_game import UniversalGame
from mcts import MCTS


# ================================================================
#  工具类
# ================================================================

class MCTSArgs:
    def __init__(self, num_mcts_sims=25, cpuct=1.0, max_depth=50):
        self.num_mcts_sims = num_mcts_sims
        self.cpuct = cpuct
        self.max_depth = max_depth

    def get(self, key, default=None):
        return getattr(self, key, default)


@st.cache_resource
def load_nnet(game_id):
    from nnet.nnet import NNetWrapper
    game = get_game_by_id(game_id)
    return NNetWrapper(game, game_id)


# ================================================================
#  棋盘渲染 (Plotly + 可点击散点)
# ================================================================

def render_board_interactive(board, n, last_move=None, valid_cells=None,
                              selected=None, valid_dests=None, is_hex=False):
    """
    渲染 Plotly 棋盘，所有格子都有可点击的散点 (customdata=[r,c])。
    保留原版棋盘外观 + 新增交互层。
    """
    fig = go.Figure()

    piece_sz = max(22, min(48, 360 // n))

    # --- 正方形棋盘底色 ---
    if not is_hex:
        checkerboard = np.zeros((n, n))
        for r in range(n):
            for c in range(n):
                checkerboard[r][c] = 0.05 if (r + c) % 2 == 0 else -0.05
        fig.add_trace(go.Heatmap(
            z=checkerboard,
            colorscale=[[0, '#2d5016'], [0.5, '#3a6b1e'], [1, '#4a8028']],
            showscale=False, hoverinfo='skip',
        ))

    # --- 计算集合 ---
    valid_set = set(valid_cells) if valid_cells else set()
    dest_set = set(valid_dests.keys()) if valid_dests else set()

    # --- 单一交互散点层 (所有格子) ---
    xs, ys = [], []
    colors, sizes, opacities, symbols, borders, border_w = ([] for _ in range(6))
    customdata = []

    for r in range(n):
        for c in range(n):
            x = (c + r * 0.5) if is_hex else c
            y = (-r * math.sqrt(3) / 2) if is_hex else r
            xs.append(x)
            ys.append(y)
            customdata.append([r, c])

            val = board[r][c]
            is_last = (last_move == (r, c))
            sym = 'hexagon' if is_hex else 'circle'

            if selected == (r, c):
                # 被选中的棋子
                colors.append('#d4a017')
                sizes.append(piece_sz)
                opacities.append(1.0)
                symbols.append(sym)
                borders.append('#ffff00')
                border_w.append(3)
            elif val == 1:
                colors.append('white')
                sizes.append(piece_sz)
                opacities.append(1.0)
                symbols.append(sym)
                borders.append('#ff8c00' if is_last else '#888')
                border_w.append(3 if is_last else 2)
            elif val == -1:
                colors.append('#1a1a2e')
                sizes.append(piece_sz)
                opacities.append(1.0)
                symbols.append(sym)
                borders.append('#ff8c00' if is_last else '#555')
                border_w.append(3 if is_last else 2)
            elif (r, c) in dest_set:
                colors.append('rgba(0,220,0,0.6)')
                sizes.append(piece_sz * 0.55)
                opacities.append(0.85)
                symbols.append(sym)
                borders.append('#0f0')
                border_w.append(2)
            elif (r, c) in valid_set:
                colors.append('rgba(100,200,100,0.5)')
                sizes.append(piece_sz * 0.5)
                opacities.append(0.75)
                symbols.append(sym)
                borders.append('#7c7')
                border_w.append(2)
            else:
                # 空格: 透明但可点击
                if is_hex:
                    colors.append('#c0c0c0')
                    sizes.append(piece_sz * 0.85)
                    opacities.append(0.4)
                    symbols.append('hexagon')
                    borders.append('#999')
                    border_w.append(1)
                else:
                    colors.append('rgba(0,0,0,0)')
                    sizes.append(piece_sz)
                    opacities.append(0.02)
                    symbols.append('square')
                    borders.append('rgba(0,0,0,0)')
                    border_w.append(0)

    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode='markers',
        marker=dict(
            size=sizes, color=colors, opacity=opacities,
            symbol=symbols,
            line=dict(width=border_w, color=borders),
        ),
        customdata=customdata,
        hoverinfo='text',
        hovertext=[f'({cd[0]},{cd[1]})' for cd in customdata],
        showlegend=False,
    ))

    # --- Hex 边框标注 ---
    if is_hex:
        fig.add_trace(go.Scatter(
            x=[c for c in range(n)], y=[0.4] * n,
            mode='lines', line=dict(color='white', width=3),
            hoverinfo='skip', showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=[c + (n-1)*0.5 for c in range(n)],
            y=[-(n-1)*math.sqrt(3)/2 - 0.4] * n,
            mode='lines', line=dict(color='white', width=3),
            hoverinfo='skip', showlegend=False,
        ))

    # --- 布局 ---
    if is_hex:
        fig.update_layout(
            width=min(600, 60*n+200), height=min(500, 50*n+200),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                       scaleanchor='x', scaleratio=1),
        )
    else:
        chart_sz = max(300, min(520, 68 * n))
        fig.update_layout(
            width=chart_sz, height=chart_sz,
            xaxis=dict(range=[-0.5, n-0.5], dtick=1, showgrid=True,
                       gridcolor='rgba(100,100,100,0.3)', zeroline=False,
                       side='top'),
            yaxis=dict(range=[n-0.5, -0.5], dtick=1, showgrid=True,
                       gridcolor='rgba(100,100,100,0.3)', zeroline=False,
                       autorange=False),
        )

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=30, r=30, t=30, b=30),
        showlegend=False,
        dragmode=False,
    )
    return fig


# ================================================================
#  对弈逻辑
# ================================================================

def execute_human_move(game, raw_board, player, action):
    if hasattr(game, 'action_mode') and game.action_mode == 'move_from_to':
        canonical = game.get_canonical_form(raw_board, player)
        new_canonical, _ = game.get_next_state(canonical, action, 1)
        new_raw = game.get_canonical_form(new_canonical, player)
        return new_raw, -player
    else:
        new_board, next_player = game.get_next_state(raw_board, action, player)
        return new_board, next_player


def execute_ai_move(game, raw_board, player, nnet, mcts_args):
    canonical = game.get_canonical_form(raw_board, player)
    mcts_engine = MCTS(game, nnet, mcts_args)
    pi = mcts_engine.getActionProb(canonical, temp=0)
    action = int(np.argmax(pi))

    if hasattr(game, 'action_mode') and game.action_mode == 'move_from_to':
        new_canonical, _ = game.get_next_state(canonical, action, 1)
        new_raw = game.get_canonical_form(new_canonical, player)
        src_r, src_c, vec_idx = game._id_to_move(action)
        dr, dc = game.move_vectors[vec_idx]
        last_move = (src_r + dr, src_c + dc)
    else:
        new_raw, _ = game.get_next_state(raw_board, action, player)
        r, c = divmod(action, game.n)
        last_move = (r, c)

    return new_raw, -player, last_move, action


def check_game_result(game, board):
    result = game.get_game_ended(board, 1)
    if result == 1:
        return True, "You Win!"
    elif result == -1:
        return True, "AI Wins!"
    elif result != 0:
        return True, "Draw!"
    return False, ""


def get_valid_cells_place(game, board):
    valids = game.get_valid_moves(board, 1)
    return {(i // game.n, i % game.n) for i in range(len(valids)) if valids[i]}


def get_valid_dests_from(game, canonical_board, src_r, src_c):
    valids = game.get_valid_moves(canonical_board, 1)
    dests = {}
    for vec_idx, (dr, dc) in enumerate(game.move_vectors):
        action_id = (src_r * game.n + src_c) * len(game.move_vectors) + vec_idx
        if action_id < len(valids) and valids[action_id]:
            dests[(src_r + dr, src_c + dc)] = action_id
    return dests


def get_my_pieces(board, n):
    return {(r, c) for r in range(n) for c in range(n) if board[r][c] == 1}


# ================================================================
#  棋盘交互 Fragment (局部刷新，减少闪烁)
# ================================================================

@st.fragment
def board_section(game, mcts_args):
    """棋盘 + 点击交互，包裹在 @st.fragment 中实现局部刷新。"""
    board = st.session_state.board
    player = st.session_state.player
    n = game.n if hasattr(game, 'n') else game.get_board_size()[0]

    is_hex = hasattr(game, 'geometry') and game.geometry == 'hex'
    is_place = hasattr(game, 'action_mode') and game.action_mode == 'place'
    is_move = hasattr(game, 'action_mode') and game.action_mode == 'move_from_to'

    # --- 状态显示 ---
    if st.session_state.game_over:
        st.success(st.session_state.game_message)
        if st.button("Play Again", type="primary"):
            start_new_game(st.session_state.game_id)
            st.rerun()
        return
    else:
        selected = st.session_state.get('selected_piece', None)
        if is_move and selected:
            st.info(f"Selected ({selected[0]},{selected[1]}) — click destination")
        else:
            st.info("Your turn (W) — click on the board" if player == 1 else "AI thinking...")

    # --- 计算合法动作 ---
    valid_cells = None
    valid_dests = None
    if player == 1:
        if is_place:
            valid_cells = get_valid_cells_place(game, board)
        elif is_move and selected:
            canonical = game.get_canonical_form(board, player)
            valid_dests = get_valid_dests_from(game, canonical, selected[0], selected[1])

    # --- 渲染棋盘 ---
    fig = render_board_interactive(
        board, n, st.session_state.last_move,
        valid_cells, selected, valid_dests, is_hex)

    if is_hex:
        st.caption("Hex: W connects top↔bottom, B connects left↔right")

    # 使用版本号作为 key，每次落子后重置选择状态
    version = st.session_state.get('board_version', 0)

    event = st.plotly_chart(
        fig,
        on_select="rerun",
        key=f"board_v{version}",
        selection_mode="points",
        width="content",
    )

    # --- 处理点击事件 ---
    if (event and hasattr(event, 'selection') and event.selection
            and event.selection.points
            and player == 1 and not st.session_state.game_over):

        point = event.selection.points[0]
        # 从 customdata 获取棋盘坐标
        cd = None
        if hasattr(point, 'customdata'):
            cd = point.customdata
        elif isinstance(point, dict):
            cd = point.get('customdata')

        if cd is not None:
            r, c = int(cd[0]), int(cd[1])
            _handle_board_click(game, board, n, player, r, c,
                                is_place, is_move, valid_cells, valid_dests,
                                selected, mcts_args)

    # --- 走棋记录 ---
    if st.session_state.move_history:
        with st.expander("Move History", expanded=False):
            for i, move in enumerate(st.session_state.move_history):
                st.text(f"{i+1}. {move}")


def _handle_board_click(game, board, n, player, r, c,
                        is_place, is_move, valid_cells, valid_dests,
                        selected, mcts_args):
    """处理棋盘上的一次点击"""
    nnet = st.session_state.nnet

    if is_place and valid_cells and (r, c) in valid_cells:
        # === Place 模式: 直接落子 ===
        action = r * n + c
        new_board, new_player = execute_human_move(game, board, player, action)
        st.session_state.board = new_board
        st.session_state.player = new_player
        st.session_state.last_move = (r, c)
        st.session_state.move_history.append(f"W: ({r},{c})")
        st.session_state.board_version = st.session_state.get('board_version', 0) + 1

        ended, msg = check_game_result(game, new_board)
        if ended:
            st.session_state.game_over = True
            st.session_state.game_message = msg
        else:
            _do_ai_turn(game, new_board, -player, nnet, mcts_args, n, is_move)

        st.rerun()

    elif is_move:
        my_pieces = get_my_pieces(board, n)

        if selected == (r, c):
            # 取消选择
            st.session_state.selected_piece = None
            st.session_state.board_version = st.session_state.get('board_version', 0) + 1
            st.rerun()

        elif valid_dests and (r, c) in valid_dests:
            # === Move 模式: 执行移动 ===
            action = valid_dests[(r, c)]
            src_r, src_c = selected
            new_board, new_player = execute_human_move(game, board, player, action)

            st.session_state.board = new_board
            st.session_state.player = new_player
            st.session_state.last_move = (r, c)
            st.session_state.selected_piece = None
            st.session_state.move_history.append(f"W: ({src_r},{src_c})->({r},{c})")
            st.session_state.board_version = st.session_state.get('board_version', 0) + 1

            ended, msg = check_game_result(game, new_board)
            if ended:
                st.session_state.game_over = True
                st.session_state.game_message = msg
            else:
                _do_ai_turn(game, new_board, -player, nnet, mcts_args, n, is_move)

            st.rerun()

        elif (r, c) in my_pieces:
            # 选择棋子
            st.session_state.selected_piece = (r, c)
            st.session_state.board_version = st.session_state.get('board_version', 0) + 1
            st.rerun()


def _do_ai_turn(game, board, player, nnet, mcts_args, n, is_move):
    """人类落子后立即执行 AI 回合 (合并到同一次刷新)"""
    if nnet is None:
        nnet = load_nnet(st.session_state.game_id)
        st.session_state.nnet = nnet

    new_board, new_player, last_move, action = execute_ai_move(
        game, board, player, nnet, mcts_args)

    if is_move:
        src_r, src_c, vec_idx = game._id_to_move(action)
        dr, dc = game.move_vectors[vec_idx]
        st.session_state.move_history.append(
            f"B: ({src_r},{src_c})->({src_r+dr},{src_c+dc})")
    else:
        ar, ac = divmod(action, n)
        st.session_state.move_history.append(f"B: ({ar},{ac})")

    st.session_state.board = new_board
    st.session_state.player = new_player
    st.session_state.last_move = last_move

    ended, msg = check_game_result(game, new_board)
    if ended:
        st.session_state.game_over = True
        st.session_state.game_message = msg


# ================================================================
#  训练面板
# ================================================================

def training_panel(game, game_id, nnet):
    st.subheader("Training")
    train_iters = st.number_input("Iterations", 1, 100, 5, key="train_iters")
    train_eps = st.number_input("Episodes / iter", 1, 20, 3, key="train_eps")
    train_sims = st.number_input("MCTS sims", 5, 50, 10, key="train_sims")

    if st.button("Start Training", width="stretch", type="primary"):
        _run_training(game, game_id, nnet, train_iters, train_eps, train_sims)


def _run_training(game, game_id, nnet, num_iters, num_eps, num_sims):
    from coach import Coach
    from utils import dotdict

    checkpoint_dir = os.path.join(_project_root, 'checkpoints', game_id)
    os.makedirs(checkpoint_dir, exist_ok=True)

    args = dotdict({
        'numIters': 1,
        'numEps': num_eps,
        'tempThreshold': 10,
        'num_mcts_sims': num_sims,
        'cpuct': 1.0,
        'checkpoint': checkpoint_dir,
        'maxlenOfQueue': 10000,
    })

    coach = Coach(game, nnet, args)
    progress = st.progress(0, text="Preparing...")
    loss_log = st.empty()
    chart_data = {'iteration': [], 'policy_loss': [], 'value_loss': []}

    for i in range(1, num_iters + 1):
        progress.progress((i - 1) / num_iters,
                          text=f"Iter {i}/{num_iters}: self-play + train...")
        coach.args = dotdict(dict(args))
        coach.args.checkpoint = checkpoint_dir
        metrics = coach.learn()

        if metrics['policy_loss']:
            pi_l = metrics['policy_loss'][-1]
            v_l = metrics['value_loss'][-1]
            chart_data['iteration'].append(i)
            chart_data['policy_loss'].append(pi_l)
            chart_data['value_loss'].append(v_l)
            loss_log.text(f"Iter {i}: pi={pi_l:.4f}  v={v_l:.4f}")

    progress.progress(1.0, text="Training complete!")
    if chart_data['iteration']:
        st.line_chart(data={'Policy Loss': chart_data['policy_loss'],
                            'Value Loss': chart_data['value_loss']})
    st.success(f"Trained {num_iters} iters on {game_id}")

    load_nnet.clear()
    st.session_state.nnet = load_nnet(game_id)


# ================================================================
#  Session State
# ================================================================

def init_session_state():
    defaults = {
        'game_id': None, 'game': None, 'board': None,
        'player': 1, 'game_over': False, 'game_message': '',
        'last_move': None, 'move_history': [], 'nnet': None,
        'selected_piece': None, 'board_version': 0,
        'llm_generated_json': None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def start_new_game(game_id):
    game = get_game_by_id(game_id)
    st.session_state.game_id = game_id
    st.session_state.game = game
    st.session_state.board = game.get_initial_board()
    st.session_state.player = 1
    st.session_state.game_over = False
    st.session_state.game_message = ''
    st.session_state.last_move = None
    st.session_state.move_history = []
    st.session_state.selected_piece = None
    st.session_state.board_version = 0
    st.session_state.nnet = load_nnet(game_id)


# ================================================================
#  Main
# ================================================================

def main():
    st.set_page_config(page_title="UniversalZero", page_icon="♟", layout="wide")
    init_session_state()

    # === 侧边栏 ===
    with st.sidebar:
        st.title("UniversalZero")
        st.caption("AlphaZero Multi-Task Engine")

        st.subheader("Game Selection")
        json_games = {k: v for k, v in GAME_REGISTRY.items()
                      if isinstance(v, UniversalGame)}
        all_games = list(GAME_REGISTRY.keys())
        selected = st.selectbox("Choose a game", options=all_games,
                                index=0 if all_games else None,
                                format_func=lambda x: f"{x} ({'JSON' if x in json_games else 'class'})")
        if st.button("New Game", type="primary"):
            if selected:
                start_new_game(selected)
                st.rerun()

        st.divider()

        st.subheader("MCTS Settings")
        num_sims = st.slider("Simulations", 5, 100, 25, step=5)
        cpuct = st.slider("Exploration (cpuct)", 0.5, 3.0, 1.0, step=0.1)
        mcts_args = MCTSArgs(num_mcts_sims=num_sims, cpuct=cpuct)

        st.divider()

        if st.session_state.game is not None:
            training_panel(st.session_state.game, st.session_state.game_id,
                           st.session_state.nnet)
            st.divider()

        with st.expander("LLM Rule Generator", expanded=False):
            try:
                from rules_translator import check_service, translate_to_json, save_rule
                _llm_available = True
            except ImportError:
                _llm_available = False
                st.warning("rules_translator not available (missing requests?)")

            if _llm_available:
                ollama_ok = check_service()
                if ollama_ok:
                    st.success("Ollama: Online")
                else:
                    st.error("Ollama: Offline — start with `ollama serve`")

                nl_input = st.text_area(
                    "Describe a new game",
                    placeholder='e.g. "5x5, place pieces, 4 in a row wins"',
                    height=80)

                if st.button("Generate Rules", disabled=not ollama_ok):
                    if nl_input.strip():
                        with st.spinner("Translating with gemma3:4b..."):
                            try:
                                rule = translate_to_json(nl_input.strip(), verbose=False)
                                filepath = save_rule(rule, verbose=False)
                                rel_path = os.path.relpath(filepath, _project_root)
                                new_game = UniversalGame(rel_path)
                                new_id = rule.get('id', os.path.splitext(
                                    os.path.basename(filepath))[0])
                                if new_id not in GAME_REGISTRY:
                                    register_game(new_id, new_game)
                                st.session_state.llm_generated_json = rule
                                st.success(f"Generated: {new_id}")
                                start_new_game(new_id)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed: {e}")

    # === 主界面 ===
    if st.session_state.game is None:
        st.markdown("## Welcome to UniversalZero")
        st.markdown("Select a game from the sidebar and click **New Game** to begin.")
        st.markdown("---")
        st.markdown("### Registered Games")
        for gid, entry in GAME_REGISTRY.items():
            if isinstance(entry, UniversalGame):
                st.markdown(f"- **{gid}**: {entry.name} "
                            f"({entry.n}x{entry.n}, {entry.action_mode}, "
                            f"{entry.end_condition.get('type', '?')})")
            elif isinstance(entry, type):
                inst = entry()
                bx, by = inst.get_board_size()
                st.markdown(f"- **{gid}**: {bx}x{by} (class)")
        return

    game = st.session_state.game
    game_name = game.name if hasattr(game, 'name') else st.session_state.game_id
    st.markdown(f"### {game_name}")

    # 棋盘区域 (fragment — 局部刷新)
    board_section(game, mcts_args)

    # 游戏信息
    with st.expander("Game Info", expanded=False):
        n = game.n if hasattr(game, 'n') else game.get_board_size()[0]
        items = [f"**Game ID:** {st.session_state.game_id}", f"**Board:** {n}x{n}"]
        if hasattr(game, 'action_mode'):
            items.append(f"**Mode:** {game.action_mode}")
        if hasattr(game, 'end_condition'):
            items.append(f"**Win:** {game.end_condition.get('type', '?')}")
        if hasattr(game, 'geometry'):
            items.append(f"**Geometry:** {game.geometry}")
        st.markdown("  \n".join(items))


if __name__ == "__main__":
    main()
