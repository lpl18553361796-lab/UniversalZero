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

# --- 核心修复：无条件强制注入路径，防止 Streamlit 热重载时跳过 ---
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
for _p in [_project_root,
           os.path.join(_project_root, "games"),
           os.path.join(_project_root, "core")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import math
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from game import GAME_REGISTRY, get_game_by_id, register_game
from universal_game import UniversalGame
from mcts import MCTS


# ================================================================
#  工具类
# ================================================================

def inject_custom_css():
    st.markdown("""
    <style>
    /* 深色科技风基础背景 */
    .stApp {
        background-color: #0e1117;
        color: #c9d1d9;
        font-family: 'Inter', sans-serif;
    }
    /* 卡片高亮与毛玻璃效果 */
    .css-1r6slb0, .css-1y4p8pa {
        background: rgba(22, 27, 34, 0.6);
        border-radius: 12px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
    }
    /* 标题样式：简洁明了 */
    h1, h2, h3 {
        color: #58a6ff;
    }
    /* 按钮样式：标准扁平化 */
    .stButton>button {
        background-color: #238636;
        color: white;
        border-radius: 4px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #2ea043;
    }
    </style>
    """, unsafe_allow_html=True)

class MCTSArgs:
    def __init__(self, num_mcts_sims=25, cpuct=1.0, max_depth=50):
        self.num_mcts_sims = num_mcts_sims
        self.cpuct = cpuct
        self.max_depth = max_depth

    def get(self, key, default=None):
        return getattr(self, key, default)


@st.cache_resource
def load_nnet(game_id, model_path=None):
    from nnet.nnet import NNetWrapper
    game = get_game_by_id(game_id)
    nnet = NNetWrapper(game, game_id)
    if model_path and os.path.exists(model_path):
        folder, filename = os.path.split(model_path)
        nnet.load_checkpoint(folder=folder, filename=filename)
    return nnet


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

    # 优化棋子大小：根据 n 动态调整，确保不超出格子
    base_sz = 300 // n
    piece_sz = max(15, min(40, base_sz * 0.8))

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
                # 被选中的棋子 (移动型游戏)
                colors.append('#d4a017') # 金色高亮
                sizes.append(piece_sz)
                opacities.append(1.0)
                symbols.append(sym)
                borders.append('#ffff00')
                border_w.append(3)
            elif (r, c) in dest_set or (r, c) in valid_set:
                # Othello 风格：落点提示 (不论是落子还是移动)
                colors.append('rgba(0, 255, 0, 0.4)') # 亮绿色提示点
                sizes.append(piece_sz * 0.45)        # 小圆圈
                opacities.append(0.8)
                symbols.append('circle')
                borders.append('rgba(0, 255, 0, 0.8)')
                border_w.append(2)
            elif val == 1:
                # 白子
                colors.append('white')
                sizes.append(piece_sz)
                opacities.append(1.0)
                symbols.append(sym)
                borders.append('#ff8c00' if is_last else '#888')
                border_w.append(3 if is_last else 2)
            elif val == -1:
                # 黑子
                colors.append('#1a1a2e')
                sizes.append(piece_sz)
                opacities.append(1.0)
                symbols.append(sym)
                borders.append('#ff8c00' if is_last else '#555')
                border_w.append(3 if is_last else 2)
            else:
                colors.append('rgba(0,0,0,0)')
                sizes.append(piece_sz)
                opacities.append(0.01)
                symbols.append('circle')
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
        dragmode=False,         # 恢复为 False，确保单点有效
        clickmode='event+select',
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
    probs = mcts_engine.getActionProb(canonical, temp=0)
    action_canonical = int(np.argmax(probs))
    
    # 核心修复：将 AI 视角的动作映射回原始坐标
    if hasattr(game, 'action_from_canonical'):
        action = game.action_from_canonical(action_canonical, player)
    else:
        action = action_canonical

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
            st.rerun(scope="fragment")
        return

    selected = st.session_state.get('selected_piece', None)
    if is_move and selected:
        st.warning(f"**Selected ({selected[0]},{selected[1]})** — click destination to move")
    elif player == 1:
        st.info("Your turn (White) — click on the board")
    else:
        st.info("AI thinking...")

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

        st.rerun(scope="fragment")

    elif is_move:
        my_pieces = get_my_pieces(board, n)

        if selected == (r, c):
            # 取消选择
            st.session_state.selected_piece = None
            st.session_state.board_version = st.session_state.get('board_version', 0) + 1
            st.rerun(scope="fragment")

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

            st.rerun(scope="fragment")

        elif (r, c) in my_pieces:
            # 选择棋子
            st.session_state.selected_piece = (r, c)
            st.session_state.board_version = st.session_state.get('board_version', 0) + 1
            st.rerun(scope="fragment")


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

def training_panel(game_id, model_path=None):
    st.caption("Small demo-friendly AlphaZero training job.")
    train_iters = st.number_input("Iterations", 1, 20, 1, key="train_iters")
    train_eps = st.number_input("Episodes / iter", 1, 20, 2, key="train_eps")
    train_sims = st.number_input("MCTS sims", 1, 50, 5, key="train_sims")

    if model_path:
        st.caption(f"Continue from: `{os.path.basename(model_path)}`")
    else:
        st.caption("Initial model: random / untrained")

    if st.button("Start Training", use_container_width=True, type="primary"):
        game = get_game_by_id(game_id)
        nnet = load_nnet(game_id, model_path)
        _run_training(game, game_id, nnet, train_iters, train_eps, train_sims)


def _run_training(game, game_id, nnet, num_iters, num_eps, num_sims):
    from coach import Coach
    from utils import dotdict

    checkpoint_dir = os.path.join(_project_root, 'checkpoints', f'web_{game_id}')
    os.makedirs(checkpoint_dir, exist_ok=True)

    args = dotdict({
        'numIters': 1,
        'numEps': num_eps,
        'tempThreshold': 10,
        'num_mcts_sims': num_sims,
        'cpuct': 1.0,
        'checkpoint': checkpoint_dir,
        'maxlenOfQueue': 10000,
        'num_workers': 1,
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
        rows = metrics.get('iterations', []) if isinstance(metrics, dict) else []

        if rows:
            row = rows[-1]
            pi_l = row.get('policy_loss')
            v_l = row.get('value_loss')
            chart_data['iteration'].append(i)
            chart_data['policy_loss'].append(pi_l)
            chart_data['value_loss'].append(v_l)
            if pi_l is not None and v_l is not None:
                loss_log.text(f"Iter {i}: pi={pi_l:.4f}  v={v_l:.4f}")
            else:
                loss_log.text(f"Iter {i}: training finished")

    progress.progress(1.0, text="Training complete!")
    if chart_data['iteration']:
        st.line_chart(data={'Policy Loss': chart_data['policy_loss'],
                            'Value Loss': chart_data['value_loss']})
    best_path = os.path.join(checkpoint_dir, 'best.pth.tar')
    st.session_state.last_trained_model = {
        'game_id': game_id,
        'path': best_path,
    }
    st.success(f"Trained {num_iters} iters on {game_id}. Saved to `{best_path}`")

    load_nnet.clear()
    if st.session_state.game_id == game_id:
        st.session_state.nnet = load_nnet(game_id, best_path)


# ================================================================
#  Session State
# ================================================================

def init_session_state():
    defaults = {
        'game_id': None, 'game': None, 'board': None,
        'player': 1, 'game_over': False, 'game_message': '',
        'last_move': None, 'move_history': [], 'nnet': None,
        'selected_piece': None, 'board_version': 0,
        'llm_generated_json': None, 'model_path': None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def start_new_game(game_id, model_path=None):
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
    st.session_state.model_path = model_path
    load_nnet.clear()
    st.session_state.nnet = load_nnet(game_id, model_path)


# ================================================================
#  Main
# ================================================================

def main():
    st.set_page_config(page_title="UniversalZero", page_icon="♟", layout="wide")
    inject_custom_css()
    init_session_state()

    # === 侧边栏 ===
    with st.sidebar:
        st.title("UniversalZero")
        st.caption("AlphaZero Multi-Task Engine")

        st.subheader("Game Selection")
        # 动态读取注册表，显示所有可玩游戏，过滤掉内部调试用的 _json 后缀
        all_games = sorted([g for g in GAME_REGISTRY.keys() if '_json' not in g])
        selected = st.selectbox("Choose a game", options=all_games,
                                index=all_games.index('hex') if 'hex' in all_games else 0)

        st.subheader("Model")
        available_models = {"⬜ Untrained (Random)": None}
        seen_labels = set()
        
        # 1. 优先扫描正式模型目录 (final_models)
        final_dir = os.path.join(_project_root, 'final_models')
        if os.path.exists(final_dir) and selected:
            for f in sorted(os.listdir(final_dir), reverse=True):
                if f.endswith('.pth.tar') and (selected.lower() in f.lower()):
                    label = f.replace('.pth.tar', '').replace('_', ' ').title()
                    # 简化显示，如 Hex Transfer 60Iters
                    full_label = f"💎 {label}"
                    available_models[full_label] = os.path.join(final_dir, f)
                    seen_labels.add(label.lower())

        # 2. 备选扫描实验目录 (experiment_results)
        exp_dir = os.path.join(_project_root, 'experiment_results')
        if os.path.exists(exp_dir) and selected:
            for f in sorted(os.listdir(exp_dir), reverse=True):
                full_path = os.path.join(exp_dir, f)
                
                # 检查是否已经在正式目录中看过了
                short_f = f.replace('.pth.tar', '').lower()
                if any(seen in short_f for seen in seen_labels):
                    continue

                if f.endswith('.pth.tar') and selected.lower() in f.lower():
                    label = f.replace('.pth.tar', '').replace('expert_injected_', 'Exp ')
                    available_models[f"🧪 {label}"] = full_path
                elif os.path.isdir(full_path) and selected.lower() in f.lower():
                    best_path = os.path.join(full_path, 'best.pth.tar')
                    if os.path.exists(best_path):
                        tag = f.replace(f"{selected}_", "").replace("_", " ")
                        available_models[f"🧪 {tag} (Best)"] = best_path

        # 3. 网页端训练保存的模型
        web_ckpt = os.path.join(_project_root, 'checkpoints', f'web_{selected}', 'best.pth.tar')
        if selected and os.path.exists(web_ckpt):
            available_models["🌐 Web Training (Best)"] = web_ckpt

        last_trained = st.session_state.get('last_trained_model')
        if (
            last_trained
            and last_trained.get('game_id') == selected
            and os.path.exists(last_trained.get('path', ''))
        ):
            available_models["Just Trained"] = last_trained['path']

        model_options = list(available_models.keys())
        default_model_index = 0
        if "Just Trained" in available_models:
            default_model_index = model_options.index("Just Trained")
        else:
            for idx, label in enumerate(model_options):
                if "Web Training" in label:
                    default_model_index = idx
                    break

        selected_label = st.selectbox(
            "Load Model",
            options=model_options,
            index=default_model_index,
        )

        if st.button("New Game", type="primary"):
            if selected:
                model_path = available_models[selected_label]
                start_new_game(selected, model_path)
                st.rerun()

        st.divider()

        # LLM 规则生成器 (LM Studio 后端)
        with st.expander("\U0001f916 LLM Rule Generator", expanded=False):
            try:
                from rules_translator import check_service, translate_to_json, save_rule, BACKEND
                _llm_available = True
            except ImportError:
                _llm_available = False
                st.warning("rules_translator not available")

            if _llm_available:
                backend_label = {
                    "lmstudio": "LM Studio",
                    "openai": "OpenAI-compatible",
                }.get(BACKEND, "Ollama")
                # 允许用户确认连接地址
                from rules_translator import LMSTUDIO_BASE_URL, OLLAMA_BASE_URL, OPENAI_BASE_URL
                default_url = {
                    "lmstudio": LMSTUDIO_BASE_URL,
                    "openai": OPENAI_BASE_URL,
                }.get(BACKEND, OLLAMA_BASE_URL)
                svc_url = st.text_input("Server URL", value=default_url, key="llm_url")
                service_ok = check_service(base_url=svc_url)
                if service_ok:
                    st.success(f"{backend_label}: \U0001f7e2 Online")
                else:
                    st.error(f"{backend_label}: \U0001f534 Offline ({svc_url})")
                    st.caption("\u8bf7\u786e\u8ba4 LM Studio \u5df2\u542f\u52a8 Local Server\uff0c\u9ed8\u8ba4\u7aef\u53e3 1234")

                nl_input = st.text_area(
                    "Describe a new game",
                    placeholder='e.g. "5x5, place pieces, 4 in a row wins"',
                    height=80)

                if st.button("Generate Rules", disabled=not service_ok):
                    if nl_input.strip():
                        with st.spinner(f"Translating with {backend_label}..."):
                            try:
                                rule = translate_to_json(
                                    nl_input.strip(),
                                    base_url=svc_url,
                                    verbose=False
                                )
                                filepath = save_rule(rule, verbose=False)
                                rel_path = os.path.relpath(filepath, os.path.join(_project_root,"games"))
                                new_game = UniversalGame(rel_path)
                                new_id = rule.get('id', os.path.splitext(
                                    os.path.basename(filepath))[0])
                                if new_id not in GAME_REGISTRY:
                                    register_game(new_id, new_game)
                                st.success(f"\u2705 Generated: {new_id}")
                                start_new_game(new_id)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed: {e}")

        st.divider()

        # Lightweight browser-accessible training entry.
        # This is intended for demonstrations; full experiments still use scripts.
        with st.expander("🏋️ Training", expanded=False):
            st.caption("Train the selected game with self-play + MCTS.")
            training_panel(selected, available_models[selected_label])

        # MCTS 参数硬编码，保持 UI 极简
        mcts_args = MCTSArgs(num_mcts_sims=50, cpuct=1.0)

    # === 主界面 ===
    if st.session_state.game is None:
        st.markdown("## Willkommen bei UniversalZero")
        st.markdown("Wählen Sie im linken Menü ein Spiel aus und klicken Sie auf **New Game**, um zu starten.")
        st.markdown("---")

        # --- Funktionsübersicht (Deutsch) ---
        st.markdown("### 📖 Funktionsübersicht")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
**🎮 KI-Kampfarena**  
Spielen Sie gegen das trainierte AlphaZero-Modell.  
Wählen Sie ein Spiel und ein Modell aus dem Seitenmenü, dann klicken Sie auf *New Game*.

**🤖 LLM-Regelgenerator**  
Beschreiben Sie ein neues Spiel in natürlicher Sprache.  
Das System übersetzt die Beschreibung automatisch in ein JSON-Regelformat und erstellt ein spielbares Spiel.
""")
        with col2:
            st.markdown("""
**📊 Erfahrungstransfer-Analyse**  
Vergleicht die Konvergenzkurven zweier Trainingsansätze:  
- *Scratch*: Modell von Grund auf trainiert  
- *Transfer*: Vortrainiertes Modell auf neues Spiel übertragen  
Misst den Jumpstart- und Speedup-Effekt des Wissenstransfers.

**🏋️ Training**  
Trainieren Sie ein neues Modell direkt im Browser  
mit Self-Play und Monte-Carlo-Tree-Search (MCTS).
""")

        st.markdown("---")

        # --- Verfügbare Spiele (Dynamisch) ---
        st.markdown("### 🎲 Verfügbare Spiele")
        # 获取所有已注册的游戏，并按名称排序
        all_registered_ids = sorted(GAME_REGISTRY.keys())
        for gid in all_registered_ids:
            # 过滤掉内部生成的临时 JSON 游戏名（可选，如果想看也可以不过滤）
            if gid.endswith('_json'): continue
            
            entry = GAME_REGISTRY.get(gid)
            if entry is None:
                continue
            if isinstance(entry, UniversalGame):
                st.markdown(f"- **{gid}**: {entry.name} "
                            f"({entry.n}×{entry.n}, Modus: `{entry.action_mode}`, "
                            f"Siegbedingung: `{entry.end_condition.get('type', '?')}`)")
            elif isinstance(entry, type):
                try:
                    inst = entry()
                    bx, by = inst.get_board_size()
                    # 尝试获取类的第一行注释作为规则简述
                    doc = entry.__doc__.split('\n')[0].strip() if entry.__doc__ else "No description available."
                    st.markdown(f"- **{gid}**: {bx}×{by} — *{doc}*")
                except:
                    st.markdown(f"- **{gid}**: (Registered Class)")

        return

    game = st.session_state.game
    game_name = game.name if hasattr(game, 'name') else st.session_state.game_id
    st.markdown(f"### {game_name}")

    if st.session_state.get('model_path'):
        st.success(f"🤖 Aktives Modell: `{st.session_state.model_path}`")
    else:
        st.warning("🤖 Aktives Modell: Untrainiert (Zufallsgewichte)")

    # 棋盘区域 (fragment — 局部刷新)
    board_section(game, mcts_args)

    # 游戏信息
    with st.expander("Spielinfo", expanded=False):
        n = game.n if hasattr(game, 'n') else game.get_board_size()[0]
        items = [f"**Game ID:** {st.session_state.game_id}", f"**Brettgröße:** {n}×{n}"]
        if hasattr(game, 'action_mode'):
            items.append(f"**Modus:** {game.action_mode}")
        if hasattr(game, 'end_condition'):
            items.append(f"**Siegbedingung:** {game.end_condition.get('type', '?')}")
        if hasattr(game, 'geometry'):
            items.append(f"**Geometrie:** {game.geometry}")
        st.markdown("  \n".join(items))

if __name__ == "__main__":
    main()
