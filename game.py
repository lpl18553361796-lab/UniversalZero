class Game:
    """
    This class specifies the base Game interface.
    To define your own game, subclass this class and implement the functions below.
    """
    def __init__(self):
        pass

    def get_initial_board(self):
        """
        Returns:
            startBoard: a representation of the initial board state.
        """
        pass

    def get_board_size(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        pass

    def get_action_size(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        pass

    def get_next_state(self, board, action, player):
        """
        Input:
            board: current board
            action: action taken by current player
            player: current player (1 or -1)

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (usually -player)
        """
        pass

    def get_valid_moves(self, board, player):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.get_action_size(), 1 for moves that are valid
        """
        pass

    def get_game_ended(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.
        """
        pass

    def get_canonical_form(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When player is black, we can invert
                            the colors and return the board.
        """
        pass

    def string_representation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        pass

# --- 游戏注册表 (Game Registry) ---
GAME_REGISTRY = {}

def register_game(name, game_class):
    """
    注册一个游戏类到全局注册表中。
    """
    GAME_REGISTRY[name] = game_class

def get_game_by_id(name, **kwargs):
    """
    根据 ID 获取游戏实例。
    """
    if name not in GAME_REGISTRY:
        raise ValueError(f"Game '{name}' not found in registry. Available games: {list(GAME_REGISTRY.keys())}")
    return GAME_REGISTRY[name](**kwargs)
