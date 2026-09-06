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

    controller.start_recording(fps=60, duration=60, outcome="maunal_test")
 
    # latest_csv = "episodes/episode_20260906_150852_overcompensation_test/actions.csv"
    # controller.play_csv(latest_csv, fps=60)
    # controller.disconnect()

if __name__ == "__main__":
    main()
