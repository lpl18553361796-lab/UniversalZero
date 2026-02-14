from game import GAME_REGISTRY, get_game_by_id

def main():
    print("=== Testing Game Registry ===")
    print(f"Initial Registry (should be empty if no imports): {GAME_REGISTRY}")
    
    print("\nImporting Breakthrough...")
    import breakthrough
    print(f"Registry after breakthrough import: {list(GAME_REGISTRY.keys())}")
    
    print("\nImporting Hex...")
    import hex_game
    print(f"Registry after hex import: {list(GAME_REGISTRY.keys())}")
    
    print("\nTesting Dynamic Instantiation...")
    bg = get_game_by_id('breakthrough')
    hg = get_game_by_id('hex', n=5) # Testing kwargs passing
    
    print(f"Breakthrough instance created: {type(bg).__name__}, Board: {bg.get_board_size()}")
    print(f"Hex instance created: {type(hg).__name__}, Board: {hg.get_board_size()}")

    if 'breakthrough' in GAME_REGISTRY and 'hex' in GAME_REGISTRY:
        print("\n>>> [PASS] Registry system working correctly.")
    else:
        print("\n>>> [FAIL] Registry incomplete.")

if __name__ == "__main__":
    main()
