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

class UniversalGUI:
    def __init__(self, game_type='breakthrough'):
        pygame.init()
        self.width, self.height = 900, 700
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(f"AlphaZero Visualization - {game_type.capitalize()}")
        self.font = pygame.font.SysFont('Arial', 24)
        
        self.game_type = game_type
        
        # 1. 初始化游戏与模型路径
        if game_type == 'hex':
            self.game = HexGame(n=7)
            self.bsize = 7
            # 优先加载迁移学习后的模型，如果没有则加载普通模型
            if os.path.exists('./temp_transfer/transferred.pth.tar'):
                self.model_path = './temp_transfer/transferred.pth.tar'
            else:
                self.model_path = './temp_hex/best.pth.tar'
        else:
            self.game = BreakthroughGame()
            self.bsize = 8
            self.model_path = './temp_strong/best.pth.tar'

        self.board = self.game.get_initial_board()
        self.player = 1 # 1:Human(White), -1:AI(Black)
        self.selected = None # Breakthrough 专用: 选中的棋子
        self.message = "Your Turn (White)"
        
        # 2. 加载 AI
        print(f"[System] Initializing AI for {game_type}...")
        self.nnet = NNetWrapper(self.game)
        if os.path.exists(self.model_path):
            print(f"Loading model: {self.model_path}")
            folder = os.path.dirname(self.model_path)
            filename = os.path.basename(self.model_path)
            self.nnet.load_checkpoint(folder, filename)
        else:
            print(f"Warning: Model {self.model_path} not found. Using Random AI.")
            
        self.mcts = MCTS(self.game, self.nnet, dotdict({'num_mcts_sims': 30, 'cpuct': 1.0}))

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
        x = size * math.sqrt(3) * (c + r/2.0) + ox
        y = size * 1.5 * r + oy
        return (x, y)

    def pixel_to_hex(self, x, y, size, ox, oy):
        """屏幕像素 -> 最近的Hex坐标"""
        for r in range(self.bsize):
            for c in range(self.bsize):
                cx, cy = self.hex_to_pixel(r, c, size, ox, oy)
                if math.hypot(x-cx, y-cy) < size * 0.8:
                    return r, c
        return None

    def draw_hex_board(self):
        size = 30
        ox, oy = 150, 80
        label = self.font.render("Hex: Connect Top - Bottom", True, COLOR_TEXT)
        self.screen.blit(label, (300, 30))

        for r in range(self.bsize):
            for c in range(self.bsize):
                center = self.hex_to_pixel(r, c, size, ox, oy)
                piece = self.board[r][c]
                color = (180, 180, 180) 
                if piece == 1: color = COLOR_W
                if piece == -1: color = COLOR_B
                self.draw_hexagon(color, center, size-2)
        return size, ox, oy

    def draw_breakthrough_board(self):
        size = 70
        ox, oy = 100, 80
        label = self.font.render("Breakthrough: Move Up", True, COLOR_TEXT)
        self.screen.blit(label, (300, 30))
        
        for r in range(self.bsize):
            for c in range(self.bsize):
                rect = (ox + c*size, oy + r*size, size, size)
                bg = (235, 236, 208) if (r+c)%2==0 else (119, 149, 86)
                if self.selected == (r, c): bg = COLOR_HIGHLIGHT
                
                pygame.draw.rect(self.screen, bg, rect)
                
                piece = self.board[r][c]
                if piece != 0:
                    color = COLOR_W if piece == 1 else COLOR_B
                    center = (ox + c*size + size//2, oy + r*size + size//2)
                    pygame.draw.circle(self.screen, color, center, size//2 - 8)
                    pygame.draw.circle(self.screen, (0,0,0), center, size//2 - 8, 1)
        return size, ox, oy

    def run(self):
        while True:
            self.screen.fill(COLOR_BG)
            if self.game_type == 'hex':
                params = self.draw_hex_board()
            else:
                params = self.draw_breakthrough_board()
            
            msg = self.font.render(f"Status: {self.message}", True, COLOR_TEXT)
            self.screen.blit(msg, (50, 650))
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT: pygame.quit(); sys.exit()
                
                if self.player == 1 and event.type == pygame.MOUSEBUTTONDOWN:
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
                            if self.board[r][c] == 1: self.selected = (r, c)
                            elif self.selected:
                                r0, c0 = self.selected
                                if r == r0 - 1 and abs(c - c0) <= 1:
                                    dir_idx = (c - c0) + 1
                                    action = (r0 * 8 + c0) * 3 + dir_idx
                                    valids = self.game.get_valid_moves(self.board, 1)
                                    if valids[action]:
                                        self.execute_move(action)
                                        self.selected = None

            res = self.game.get_game_ended(self.board, 1)
            if res != 0:
                self.message = "YOU WIN!" if res == 1 else "AI WINS!"
                continue

            if self.player == -1:
                pygame.event.pump()
                self.message = "AI Thinking..."
                # 强制刷新一次界面显示 Thinking
                msg = self.font.render(f"Status: {self.message}", True, COLOR_TEXT)
                pygame.draw.rect(self.screen, COLOR_BG, (50, 650, 400, 50))
                self.screen.blit(msg, (50, 650))
                pygame.display.flip()
                
                canonical = self.game.get_canonical_form(self.board, -1)
                probs = self.mcts.getActionProb(canonical, temp=0)
                action = np.argmax(probs)
                next_s, _ = self.game.get_next_state(canonical, action, 1)
                self.board = self.game.get_canonical_form(next_s, -1)
                self.player = 1
                self.message = "Your Turn (White)"

    def execute_move(self, action):
        self.board, _ = self.game.get_next_state(self.board, action, 1)
        self.player = -1

if __name__ == "__main__":
    g_type = 'breakthrough'
    if len(sys.argv) > 1: g_type = sys.argv[1]
    gui = UniversalGUI(g_type)
    gui.run()
