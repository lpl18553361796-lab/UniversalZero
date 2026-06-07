import pygame
import sys
import numpy as np
import math
import os
from breakthrough import BreakthroughGame
from hex_game import HexGame
from nnet.nnet import NNetWrapper
from mcts import MCTS
from utils import dotdict

# --- 配色方案 ---
COLOR_BG = (40, 44, 52)       # 深灰背景
COLOR_GRID = (80, 80, 80)     # 网格线
COLOR_W = (240, 240, 240)     # 白棋
COLOR_B = (30, 30, 30)        # 黑棋
COLOR_HIGHLIGHT = (100, 200, 100) # 选中高亮
COLOR_TEXT = (255, 255, 255)
COLOR_PANEL = (30, 33, 40)    # 面板背景


class UniversalGUI:
    def __init__(self, game_type='breakthrough'):
        pygame.init()
        self.width, self.height = 1200, 750
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(f"AlphaZero Visualization - {game_type.capitalize()}")
        self.font = pygame.font.SysFont('Arial', 22)
        self.font_small = pygame.font.SysFont('Arial', 13)
        self.font_bold = pygame.font.SysFont('Arial', 20, bold=True)

        self.game_type = game_type

        # 1. 初始化游戏与模型路径
        if game_type == 'hex':
            self.game = HexGame(n=7)
            self.bsize = 7
            if os.path.exists('./temp_transfer/transferred.pth.tar'):
                self.model_path = './temp_transfer/transferred.pth.tar'
            else:
                self.model_path = './temp_hex/best.pth.tar'
        else:
            self.game = BreakthroughGame()
            self.bsize = 8
            self.model_path = './temp_strong/best.pth.tar'

        self.board = self.game.get_initial_board()
        self.player = 1  # 1:Human(White), -1:AI(Black)
        self.selected = None  # Breakthrough 专用: 选中的棋子
        self.message = "Your Turn (White)"
        self.game_over = False

        # 2. 可视化状态
        self.ai_policy_heatmap = None  # (bsize, bsize) 全局坐标热力图
        self.v_history = []            # 胜率历史 (白方视角)
        self.move_count = 0

        # 3. 加载 AI
        print(f"[System] Initializing AI for {game_type}...")
        self.nnet = NNetWrapper(self.game, game_id=game_type)
        if os.path.exists(self.model_path):
            print(f"Loading model: {self.model_path}")
            folder = os.path.dirname(self.model_path)
            filename = os.path.basename(self.model_path)
            self.nnet.load_checkpoint(folder, filename)
        else:
            print(f"Warning: Model {self.model_path} not found. Using Random AI.")

        self.mcts = MCTS(self.game, self.nnet, dotdict({'num_mcts_sims': 30, 'cpuct': 1.0}))

        # 4. 初始局面评估
        self._update_visualization()

    # =================================================================
    #  可视化数据计算
    # =================================================================

    def _update_visualization(self, mcts_probs=None):
        """更新热力图和胜率数据"""
        # AI (player -1) 的 Canonical 视角
        canonical = self.game.get_canonical_form(self.board, -1)

        # 策略：优先用 MCTS 的完整搜索结果，否则用神经网络的快速预测
        if mcts_probs is not None:
            pi = np.array(mcts_probs)
        else:
            pi, _ = self.nnet.predict(canonical)

        # 价值：始终用神经网络预测 (从白方视角)
        _, v_raw = self.nnet.predict(canonical)
        v_white = -float(v_raw[0])  # 取反 → 白方视角
        self.v_history.append(v_white)

        # 将策略映射到全局坐标热力图
        if self.game_type == 'breakthrough':
            self.ai_policy_heatmap = self._map_policy_bt(pi)
        else:
            self.ai_policy_heatmap = self._map_policy_hex(pi)

    def _map_policy_bt(self, pi):
        """Breakthrough: 将 AI canonical 策略 → 全局坐标的目标格热力图"""
        heatmap = np.zeros((8, 8))
        for action_idx in range(192):
            prob = pi[action_idx]
            if prob < 0.001:
                continue
            dir_mapped = action_idx % 3
            square = action_idx // 3
            src_r = square // 8
            src_c = square % 8
            tgt_r_canon = src_r - 1
            tgt_c_canon = src_c + (dir_mapped - 1)
            if 0 <= tgt_r_canon < 8 and 0 <= tgt_c_canon < 8:
                # Canonical → Global: 垂直翻转 (AI 的视角是翻转的)
                tgt_r_global = 7 - tgt_r_canon
                tgt_c_global = tgt_c_canon
                heatmap[tgt_r_global][tgt_c_global] += prob
        return heatmap

    def _map_policy_hex(self, pi):
        """Hex: 将 AI canonical 策略 → 全局坐标的落子热力图"""
        heatmap = np.zeros((7, 7))
        for action_idx in range(49):
            prob = pi[action_idx]
            if prob < 0.001:
                continue
            canon_r = action_idx // 7
            canon_c = action_idx % 7
            # Canonical for player -1: -board.T → 转置还原
            global_r = canon_c
            global_c = canon_r
            heatmap[global_r][global_c] += prob
        return heatmap

    # =================================================================
    #  棋盘绘制
    # =================================================================

    def draw_hexagon(self, color, center, size):
        """绘制正六边形"""
        points = []
        for i in range(6):
            angle_deg = 60 * i + 30
            angle_rad = math.radians(angle_deg)
            x = center[0] + size * math.cos(angle_rad)
            y = center[1] + size * math.sin(angle_rad)
            points.append((x, y))
        pygame.draw.polygon(self.screen, color, points)
        pygame.draw.polygon(self.screen, COLOR_GRID, points, 2)

    def hex_to_pixel(self, r, c, size, ox, oy):
        """Hex坐标 -> 屏幕像素 (Pointy-topped)"""
        x = size * math.sqrt(3) * (c + r / 2.0) + ox
        y = size * 1.5 * r + oy
        return (x, y)

    def pixel_to_hex(self, x, y, size, ox, oy):
        """屏幕像素 -> 最近的Hex坐标"""
        for r in range(self.bsize):
            for c in range(self.bsize):
                cx, cy = self.hex_to_pixel(r, c, size, ox, oy)
                if math.hypot(x - cx, y - cy) < size * 0.8:
                    return r, c
        return None

    def draw_hex_board(self):
        size = 30
        ox, oy = 150, 100
        label = self.font.render("Hex: Connect Top - Bottom", True, COLOR_TEXT)
        self.screen.blit(label, (200, 50))

        for r in range(self.bsize):
            for c in range(self.bsize):
                center = self.hex_to_pixel(r, c, size, ox, oy)
                piece = self.board[r][c]
                color = (180, 180, 180)
                if piece == 1:
                    color = COLOR_W
                if piece == -1:
                    color = COLOR_B
                self.draw_hexagon(color, center, size - 2)
        return size, ox, oy

    def draw_breakthrough_board(self):
        size = 70
        ox, oy = 50, 80
        label = self.font.render("Breakthrough: Move Up", True, COLOR_TEXT)
        self.screen.blit(label, (200, 40))

        for r in range(self.bsize):
            for c in range(self.bsize):
                rect = (ox + c * size, oy + r * size, size, size)
                bg = (235, 236, 208) if (r + c) % 2 == 0 else (119, 149, 86)
                if self.selected == (r, c):
                    bg = COLOR_HIGHLIGHT

                pygame.draw.rect(self.screen, bg, rect)

                piece = self.board[r][c]
                if piece != 0:
                    color = COLOR_W if piece == 1 else COLOR_B
                    center = (ox + c * size + size // 2, oy + r * size + size // 2)
                    pygame.draw.circle(self.screen, color, center, size // 2 - 8)
                    pygame.draw.circle(self.screen, (0, 0, 0), center, size // 2 - 8, 1)
        return size, ox, oy

    # =================================================================
    #  策略热力图绘制
    # =================================================================

    def _draw_heatmap_bt(self, size, ox, oy):
        """在 Breakthrough 棋盘上叠加策略热力图"""
        if self.ai_policy_heatmap is None:
            return
        max_val = np.max(self.ai_policy_heatmap)
        if max_val < 0.001:
            return

        for r in range(8):
            for c in range(8):
                val = self.ai_policy_heatmap[r][c]
                if val < 0.005:
                    continue
                intensity = val / max_val

                # 半透明叠加层
                surf = pygame.Surface((size, size), pygame.SRCALPHA)
                alpha = int(intensity * 140)
                # 渐变色: 低=淡黄, 高=亮红
                red = 255
                green = int(200 * (1 - intensity))
                surf.fill((red, green, 0, alpha))
                self.screen.blit(surf, (ox + c * size, oy + r * size))

                # 概率百分比标注
                if intensity > 0.15:
                    pct = f"{val * 100:.0f}%"
                    txt = self.font_small.render(pct, True, (255, 255, 255))
                    tx = ox + c * size + size // 2 - txt.get_width() // 2
                    ty = oy + r * size + size - 16
                    self.screen.blit(txt, (tx, ty))

    def _draw_heatmap_hex(self, size, ox, oy):
        """在 Hex 棋盘上叠加策略热力图"""
        if self.ai_policy_heatmap is None:
            return
        max_val = np.max(self.ai_policy_heatmap)
        if max_val < 0.001:
            return

        for r in range(7):
            for c in range(7):
                val = self.ai_policy_heatmap[r][c]
                if val < 0.005:
                    continue
                # 只在空位上叠加
                if self.board[r][c] != 0:
                    continue

                intensity = val / max_val
                center = self.hex_to_pixel(r, c, size, ox, oy)

                radius = int((size - 2) * 0.75)
                surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                alpha = int(intensity * 160)
                red = 255
                green = int(180 * (1 - intensity))
                pygame.draw.circle(surf, (red, green, 0, alpha), (radius, radius), radius)
                self.screen.blit(surf, (int(center[0]) - radius, int(center[1]) - radius))

                # 概率标注
                if intensity > 0.15:
                    pct = f"{val * 100:.0f}%"
                    txt = self.font_small.render(pct, True, (255, 255, 255))
                    self.screen.blit(txt, (int(center[0]) - txt.get_width() // 2,
                                           int(center[1]) - txt.get_height() // 2))

    # =================================================================
    #  胜率波动图
    # =================================================================

    def _draw_value_chart(self):
        """在右侧面板绘制胜率实时曲线"""
        # 图表区域
        cx, cy = 700, 100
        cw, ch = 460, 480
        padding = 25

        # 面板背景
        pygame.draw.rect(self.screen, COLOR_PANEL, (cx - 45, cy - 40, cw + 55, ch + 90))
        pygame.draw.rect(self.screen, (70, 70, 70), (cx - 45, cy - 40, cw + 55, ch + 90), 1)

        # 标题
        title = self.font_bold.render("Evaluation (White POV)", True, COLOR_TEXT)
        self.screen.blit(title, (cx, cy - 30))

        # 绘图区
        plot_x = cx + padding
        plot_y = cy + 10
        plot_w = cw - 2 * padding
        plot_h = ch - 40
        mid_y = plot_y + plot_h // 2

        # 网格线 + Y轴标签
        for frac, label in [(1.0, "+1.0"), (0.5, "+0.5"), (0.0, " 0.0"),
                            (-0.5, "-0.5"), (-1.0, "-1.0")]:
            y = mid_y - int(frac * plot_h / 2)
            line_color = (100, 100, 100) if frac == 0 else (55, 55, 55)
            line_width = 2 if frac == 0 else 1
            pygame.draw.line(self.screen, line_color,
                             (plot_x, y), (plot_x + plot_w, y), line_width)
            lbl = self.font_small.render(label, True, (140, 140, 140))
            self.screen.blit(lbl, (plot_x - 38, y - 7))

        # "White +" / "Black +" 标签
        w_lbl = self.font_small.render("White +", True, (100, 220, 100))
        self.screen.blit(w_lbl, (plot_x + plot_w + 5, plot_y))
        b_lbl = self.font_small.render("Black +", True, (255, 100, 100))
        self.screen.blit(b_lbl, (plot_x + plot_w + 5, plot_y + plot_h - 14))

        # 无数据时显示提示
        if len(self.v_history) < 1:
            no_data = self.font_small.render("Waiting for moves...", True, (100, 100, 100))
            self.screen.blit(no_data, (plot_x + plot_w // 2 - 50, mid_y - 7))
            return

        # 构建坐标点
        n = len(self.v_history)
        if n == 1:
            x_step = 0
        else:
            x_step = plot_w / (n - 1)

        points = []
        for i, v in enumerate(self.v_history):
            px = plot_x + int(i * x_step)
            py = mid_y - int(v * plot_h / 2)
            py = max(plot_y + 2, min(plot_y + plot_h - 2, py))
            points.append((px, py))

        # 填充正负区域 (半透明)
        if len(points) >= 2:
            # 正值区域 (绿色) + 负值区域 (红色)
            for i in range(len(points) - 1):
                x1, y1 = points[i]
                x2, y2 = points[i + 1]
                # 构建四边形: 两个数据点 + 零线上的两个对应点
                zero_y1 = mid_y
                zero_y2 = mid_y
                polygon = [(x1, y1), (x2, y2), (x2, zero_y2), (x1, zero_y1)]

                avg_v = (self.v_history[i] + self.v_history[i + 1]) / 2
                surf = pygame.Surface((abs(x2 - x1) + 1, plot_h), pygame.SRCALPHA)
                if avg_v >= 0:
                    fill_color = (60, 200, 60, 35)
                else:
                    fill_color = (200, 60, 60, 35)

                # 简化: 直接画半透明矩形覆盖区域
                region_top = min(y1, y2, mid_y)
                region_bot = max(y1, y2, mid_y)
                region_surf = pygame.Surface((abs(x2 - x1) + 1, region_bot - region_top + 1),
                                             pygame.SRCALPHA)
                region_surf.fill(fill_color)
                self.screen.blit(region_surf, (min(x1, x2), region_top))

        # 绘制曲线
        if len(points) >= 2:
            pygame.draw.lines(self.screen, (0, 180, 255), False, points, 2)

        # 绘制数据点
        for i, (px, py) in enumerate(points):
            v = self.v_history[i]
            dot_color = (100, 255, 100) if v >= 0 else (255, 100, 100)
            pygame.draw.circle(self.screen, dot_color, (px, py), 4)
            pygame.draw.circle(self.screen, (255, 255, 255), (px, py), 4, 1)

        # 当前评估值标签
        last_v = self.v_history[-1]
        v_color = (100, 255, 100) if last_v >= 0 else (255, 100, 100)
        v_label = self.font.render(f"v = {last_v:+.3f}", True, v_color)
        self.screen.blit(v_label, (cx + 10, cy + ch + 10))

        # 步数标签
        moves_label = self.font_small.render(f"Move {n - 1}", True, (140, 140, 140))
        self.screen.blit(moves_label, (cx + cw - 60, cy + ch + 15))

    # =================================================================
    #  热力图图例
    # =================================================================

    def _draw_heatmap_legend(self):
        """在棋盘下方绘制热力图图例"""
        lx, ly = 50, 660
        label = self.font_small.render("AI Policy Heatmap:", True, (180, 180, 180))
        self.screen.blit(label, (lx, ly))

        # 渐变条
        bar_x = lx + 130
        bar_w = 200
        bar_h = 14
        for i in range(bar_w):
            intensity = i / bar_w
            red = 255
            green = int(200 * (1 - intensity))
            pygame.draw.line(self.screen, (red, green, 0),
                             (bar_x + i, ly + 2), (bar_x + i, ly + bar_h + 2))
        pygame.draw.rect(self.screen, (100, 100, 100), (bar_x, ly + 2, bar_w, bar_h), 1)

        # 标签
        low = self.font_small.render("Low", True, (140, 140, 140))
        high = self.font_small.render("High", True, (140, 140, 140))
        self.screen.blit(low, (bar_x - 5, ly + 18))
        self.screen.blit(high, (bar_x + bar_w - 15, ly + 18))

    # =================================================================
    #  主循环
    # =================================================================

    def run(self):
        clock = pygame.time.Clock()

        while True:
            clock.tick(30)
            self.screen.fill(COLOR_BG)

            # --- 绘制棋盘 ---
            if self.game_type == 'hex':
                params = self.draw_hex_board()
            else:
                params = self.draw_breakthrough_board()

            # --- 叠加策略热力图 ---
            if not self.game_over:
                if self.game_type == 'hex':
                    self._draw_heatmap_hex(*params)
                else:
                    self._draw_heatmap_bt(*params)

            # --- 绘制胜率曲线 ---
            self._draw_value_chart()

            # --- 绘制热力图图例 ---
            self._draw_heatmap_legend()

            # --- 状态信息 ---
            msg = self.font.render(f"Status: {self.message}", True, COLOR_TEXT)
            self.screen.blit(msg, (50, 700))
            pygame.display.flip()

            # --- 事件处理 ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if self.player == 1 and not self.game_over and event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = pygame.mouse.get_pos()

                    if self.game_type == 'hex':
                        res = self.pixel_to_hex(mx, my, *params)
                        if res:
                            r, c = res
                            if self.board[r][c] == 0:
                                action = r * self.bsize + c
                                self.execute_move(action)
                    else:
                        size, ox, oy = params
                        c = (mx - ox) // size
                        r = (my - oy) // size
                        if 0 <= r < 8 and 0 <= c < 8:
                            if self.board[r][c] == 1:
                                self.selected = (r, c)
                            elif self.selected:
                                r0, c0 = self.selected
                                if r == r0 - 1 and abs(c - c0) <= 1:
                                    dir_idx = (c - c0) + 1
                                    action = (r0 * 8 + c0) * 3 + dir_idx
                                    valids = self.game.get_valid_moves(self.board, 1)
                                    if valids[action]:
                                        self.execute_move(action)
                                        self.selected = None

            # --- 检查游戏结束 ---
            res = self.game.get_game_ended(self.board, 1)
            if res != 0:
                if not self.game_over:
                    self.game_over = True
                    self.message = "YOU WIN!" if res == 1 else "AI WINS!"
                continue

            # --- AI 回合 ---
            if self.player == -1 and not self.game_over:
                pygame.event.pump()
                self.message = "AI Thinking..."
                # 强制刷新界面显示 Thinking
                pygame.draw.rect(self.screen, COLOR_BG, (50, 700, 500, 40))
                msg = self.font.render(f"Status: {self.message}", True, COLOR_TEXT)
                self.screen.blit(msg, (50, 700))
                pygame.display.flip()

                canonical = self.game.get_canonical_form(self.board, -1)
                probs = self.mcts.getActionProb(canonical, temp=0)
                action = np.argmax(probs)
                next_s, _ = self.game.get_next_state(canonical, action, 1)
                self.board = self.game.get_canonical_form(next_s, -1)
                self.player = 1
                self.move_count += 1
                self.message = "Your Turn (White)"

                # 更新可视化 (使用 MCTS 搜索的完整策略)
                self._update_visualization(mcts_probs=np.array(probs))

    def execute_move(self, action):
        self.board, _ = self.game.get_next_state(self.board, action, 1)
        self.player = -1
        self.move_count += 1
        # 更新可视化 (使用神经网络快速预测)
        self._update_visualization()


if __name__ == "__main__":
    g_type = 'breakthrough'
    if len(sys.argv) > 1:
        g_type = sys.argv[1]
    gui = UniversalGUI(g_type)
    gui.run()
