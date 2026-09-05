import time
from wrapper import Portal2Controller, is_game_running, launch_game

def main():
    port = 8020
    
    # Standard startup checks
    if not is_game_running():
        if not launch_game(port):
            return
            
    controller = Portal2Controller(port)
    
    # Try connecting
    connected = False
    for attempt in range(5):
        if controller.connect():
            connected = True
            break
        print(f"Retrying connection ({attempt + 1}/5)...")
        time.sleep(3)
        
    if not connected:
        print("Could not connect to the game console.")
        return
        
    # Clear initial connection spam
    time.sleep(0.5)
    controller.read_console(print_to_terminal=False)

    print("\n--- Testing Custom Sequence ---")
    
    # Load the test map and wait for physics
    controller.load_map("puzzlemaker/preview", wait_for_load=True)
    
    print("\n>>> MAP LOADED! <<<")
    print(">>> Automatically verifying player control by testing jumps... <<<")
    
    # Establish a baseline position and angle
    baseline_pos = None
    while not baseline_pos:
        controller.send_command("getpos")
        time.sleep(0.5)
        out = controller.read_console(print_to_terminal=False)
        for line in out.split('\n'):
            if "setpos" in line:
                baseline_pos = line.strip()
                break
                
    # Poll until the jump successfully changes the position
    control_verified = False
    while not control_verified:
        controller.jump()
        time.sleep(0.3) 
        controller.send_command("getpos")
        time.sleep(0.2)
        out = controller.read_console(print_to_terminal=False)
        
        current_pos = baseline_pos
        for line in reversed(out.split('\n')):
            if "setpos" in line:
                current_pos = line.strip()
                break
                
        if current_pos != baseline_pos:
            control_verified = True
        else:
            print("Jump did not affect position (player likely frozen or fading in). Retrying...")
            baseline_pos = current_pos
            time.sleep(1.0)
            
    print("\nPlayer has active control. Starting the specific movement sequence in 1 second...")
    time.sleep(1.0)
    
    # --- The Sequence ---
    
    # 1. Walk forward for 2 seconds
    print("1. Walking forward (2s)...")
    controller.move_forward(True)
    time.sleep(2.0)
    controller.move_forward(False)
    time.sleep(0.5)
    
    # 2. Grab
    print("2. Grabbing (interact)...")
    controller.interact()
    time.sleep(0.5)
    
    # 3. Turn ~45 degrees right
    # (Note: Exact degrees depend on your in-game mouse sensitivity. This simulates a continuous sweep.)
    print("3. Turning right (~45 degrees)...")
    controller.move_mouse(dx=20, dy=0, steps=15, delay=0.01)
    time.sleep(0.5)
    
    # 4. Walk a little forward
    print("4. Walking forward (1s)...")
    controller.move_forward(True)
    time.sleep(1.0)
    controller.move_forward(False)
    time.sleep(0.5)
    
    # 5. Drop
    print("5. Dropping object...")
    controller.interact()
    time.sleep(0.5)
    
    # 6. Turn ~45 degrees left (back to original heading)
    print("6. Turning left (~45 degrees)...")
    controller.move_mouse(dx=-20, dy=0, steps=15, delay=0.01)
    time.sleep(0.5)
    
    # 7. Walk forward
    print("7. Walking forward (2s)...")
    controller.move_forward(True)
    time.sleep(2.0)
    controller.move_forward(False)
    
    print("\n--- Sequence Complete ---")
    controller.disconnect()

if __name__ == "__main__":
    main()
