import time
from wrapper import Portal2Controller, launch_game, is_game_running

def main():
    port = 8020
    if not is_game_running():
        launch_game(port)
        
    controller = Portal2Controller(port)
    if not controller.connect():
        print("Failed to connect.")
        return

    print("\n--- Testing Programmatic Recording ---")
    print("Recording will start in 3 seconds. Move around!")
    time.sleep(3)
    
    # # Start recording manually
    # controller.start_recording(fps=60)
    
    # # Record for 5 seconds
    # time.sleep(40)
    
    # # Stop recording manually
    # controller.stop_recording(outcome="overcompensation_test")
    
    # Note: Get the latest CSV file path from your 'episodes' folder
    latest_csv = "episodes/episode_20260906_150852_overcompensation_test/actions.csv"
    
    # Example Playback:
    print("\n--- Testing Playback ---")
    print("Hands off the keyboard! Playing back in 3 seconds...")
    time.sleep(3)
    controller.play_csv(latest_csv, fps=60)
    
    controller.disconnect()

if __name__ == "__main__":
    main()
