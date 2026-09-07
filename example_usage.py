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

    # controller.start_recording(fps=60, duration=30, outcome="laptop_test")
 
    controller.play_csv(csv_path="sequences/TEST_0_mimic_sequence/actions.csv", fps=60)
    controller.disconnect()


    time.sleep(0.5)

if __name__ == "__main__":
    main()
