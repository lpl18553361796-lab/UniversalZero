"""
UniversalZero — Streamlit Heatmap Visualization v1
功能：在原有 app.py 基础上增加 AI 策略热力图展示，
直观显示 AI 认为哪些位置的落子概率更高（直觉强度）。
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
import torch

from game import GAME_REGISTRY, get_game_by_id
from mcts import MCTS
from app import load_nnet, inject_custom_css, init_session_state, start_new_game, MCTSArgs, _handle_board_click

# ================================================================
#  热力图辅助函数
# ================================================================

def get_ai_policy_heatmap(game, board, player, nnet):
    """获取 AI 的原始策略概率分布 (不经过 MCTS)"""
    canonical = game.get_canonical_form(board, player)
    nnet.nnet.eval()
    with torch.no_grad():
        # 获取预测结果
        pi, v = nnet.predict(canonical)
        # pi 已经是 log_softmax 之后的概率 (exp 后的)
    
    # 将一维概率还原为棋盘二维形状
    n = game.n
    probs = pi.reshape(n, n)
    
    # 如果是黑方，记得把概率也“转置”回来以匹配原始坐标系
    if player == -1 and hasattr(game, 'geometry') and game.geometry == 'hex':
        probs = probs.T
        
    return probs, v

# ================================================================
#  增强版棋盘渲染 (带热力图)
# ================================================================

def render_board_with_heatmap(board, n, probs, last_move=None, is_hex=False):
    fig = go.Figure()
    piece_sz = max(22, min(48, 360 // n))

    # --- 1. 底层：策略热力图 ---
    xs_bg, ys_bg = [], []
    for r in range(n):
        for c in range(n):
            x = (c + r * 0.5) if is_hex else c
            y = (-r * math.sqrt(3) / 2) if is_hex else r
            xs_bg.append(x)
            ys_bg.append(y)

    # 绘制热力图色块
    fig.add_trace(go.Heatmap(
        x=xs_bg if not is_hex else None, # Hex 坐标复杂，用 Scatter 模拟热力图
        y=ys_bg if not is_hex else None,
        z=probs,
        colorscale='Viridis',
        opacity=0.3,
        showscale=True,
        hoverinfo='skip',
        name='AI Policy Intuition'
    ))

    # --- 2. 中层：棋子层 (复用之前的逻辑) ---
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

            if val == 1:
                colors.append('white'); sizes.append(piece_sz); opacities.append(1.0)
                symbols.append(sym); borders.append('#ff8c00' if is_last else '#888'); border_w.append(3 if is_last else 2)
            elif val == -1:
                colors.append('#1a1a2e'); sizes.append(piece_sz); opacities.append(1.0)
                symbols.append(sym); borders.append('#ff8c00' if is_last else '#555'); border_w.append(3 if is_last else 2)
            else:
                # 空位显示微弱的热力图交互点
                colors.append('rgba(0,0,0,0)'); sizes.append(piece_sz); opacities.append(0.01)
                symbols.append('square'); borders.append('rgba(0,0,0,0)'); border_w.append(0)

    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode='markers',
        marker=dict(size=sizes, color=colors, opacity=opacities, symbol=symbols,
                    line=dict(width=border_w, color=borders)),
        customdata=customdata,
        hoverinfo='text',
        hovertext=[f'Prob: {probs[cd[0],cd[1]]:.4f}' for cd in customdata],
        showlegend=False,
    ))

    # --- 布局配置 ---
    fig.update_layout(
        width=700, height=600,
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor='x', scaleratio=1),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig

# ================================================================
#  Fragment
# ================================================================

@st.fragment
def heatmap_board_section(game, mcts_args):
    board = st.session_state.board
    player = st.session_state.player
    nnet = st.session_state.nnet
    n = game.n
    
    if nnet is None:
        st.warning("Please load a model first.")
        return

    # 1. 计算热力图
    probs, value = get_ai_policy_heatmap(game, board, player, nnet)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.write(f"AI Win Rate Estimate: **{((value[0]+1)/2)*100:.1f}%**")
        fig = render_board_with_heatmap(board, n, probs, st.session_state.last_move, is_hex=True)
        
        version = st.session_state.get('board_version', 0)
        event = st.plotly_chart(fig, on_select="rerun", key=f"heatmap_v{version}", selection_mode="points")
        
        # 处理点击
        if (event and hasattr(event, 'selection') and event.selection and event.selection.points 
            and player == 1 and not st.session_state.game_over):
            point = event.selection.points[0]
            cd = point.get('customdata') if isinstance(point, dict) else getattr(point, 'customdata', None)
            if cd:
                r, c = int(cd[0]), int(cd[1])
                _handle_board_click(game, board, n, player, r, c, True, False, None, None, None, mcts_args)
                st.rerun(scope="fragment")

    with col2:
        st.subheader("Top Suggestions")
        flat_probs = probs.flatten()
        top_indices = np.argsort(flat_probs)[-5:][::-1]
        for idx in top_indices:
            r, c = divmod(idx, n)
            st.write(f"({r},{c}): {flat_probs[idx]:.2%}")

# ================================================================
#  Main
# ================================================================

def main():
    st.set_page_config(page_title="UniversalZero Heatmap", page_icon="🔥", layout="wide")
    inject_custom_css()
    init_session_state()

    with st.sidebar:
        st.title("Intuition Visualizer")
        all_games = sorted([g for g in GAME_REGISTRY.keys() if '_json' not in g])
        selected = st.selectbox("Game", options=all_games, index=all_games.index('hex') if 'hex' in all_games else 0)
        
        # 模型选择逻辑 (包含我们的注入模型)
        available_models = {"Random": None}
        exp_dir = os.path.join(_project_root, 'experiment_results')
        if os.path.exists(exp_dir):
            for f in os.listdir(exp_dir):
                full_path = os.path.join(exp_dir, f)
                # 场景 1: 直接在目录下的模型文件 (如 expert_injected_othello.pth.tar)
                if f.endswith('.pth.tar'):
                    available_models[f.replace('.pth.tar', '')] = full_path
                # 场景 2: 子文件夹下的 best.pth.tar (如 results_transfer_frozen/best.pth.tar)
                elif os.path.isdir(full_path):
                    best_path = os.path.join(full_path, 'best.pth.tar')
                    if os.path.exists(best_path):
                        available_models[f] = best_path
        
        # 也要显示正在训练的结果文件夹 (如果有 best.pth.tar 产生)
        res_dir = os.path.join(_project_root, 'results_transfer_frozen')
        if os.path.exists(res_dir):
            path = os.path.join(res_dir, 'best.pth.tar')
            if os.path.exists(path): available_models["Training (Frozen)"] = path

        sel_model = st.selectbox("Model", options=list(available_models.keys()))
        
        if st.button("Start Game", type="primary"):
            start_new_game(selected, available_models[sel_model])
            st.rerun()

    if st.session_state.game:
        heatmap_board_section(st.session_state.game, MCTSArgs(num_mcts_sims=50))
    else:
        st.info("Select a game and model to begin visualization.")

if __name__ == "__main__":
    main()
