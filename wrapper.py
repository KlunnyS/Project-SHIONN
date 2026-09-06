import subprocess
import socket
import time
import sys

class Portal2Controller:
    """Handles communication and movement commands for Portal 2 via the developer console network port."""
    def __init__(self, port=8020, log_file="portal2_console.log"):
        self.port = port
        self.sock = None
        self.ui = None
        self.recorder = None
        self.log_file = log_file
        self.event_buffer = []  # Stores recent console lines for I/O checking
        # Clear the log file at startup
        if self.log_file:
            with open(self.log_file, 'w') as f:
                f.write("--- Portal 2 Console Log ---\n")

    def connect(self):
        """Attempts to connect to the Portal 2 netconport."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(2.0)
        try:
            self.sock.connect(("127.0.0.1", self.port))
            print(f"Connected to Portal 2 console on port {self.port}.")
            # Make the socket non-blocking for reading
            self.sock.setblocking(False)
            return True
        except Exception as e:
            print(f"Failed to connect to Portal 2: {e}")
            return False

    def send_command(self, cmd):
        """Sends a raw command to the developer console."""
        if self.sock:
            try:
                self.sock.sendall((cmd + "\n").encode('utf-8'))
            except Exception as e:
                print(f"Error sending command '{cmd}': {e}")

    def read_console(self, print_to_terminal=False):
        """Reads available output from the developer console and logs it."""
        if not self.sock:
            return ""
        
        output = ""
        try:
            while True:
                data = self.sock.recv(4096)
                if not data:
                    break
                decoded = data.decode('utf-8', errors='ignore')
                output += decoded
                # Optionally print to terminal
                if print_to_terminal:
                    sys.stdout.write(decoded)
                    sys.stdout.flush()
        except BlockingIOError:
            # No more data available to read right now
            pass
        except Exception as e:
            print(f"Error reading console: {e}")
            
        if output:
            self.event_buffer.extend(output.splitlines())
            # Keep buffer from growing infinitely
            if len(self.event_buffer) > 1000:
                self.event_buffer = self.event_buffer[-1000:]
            
        if output and self.log_file:
            import datetime
            ts = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S.%f] ")
            # Prefix every line with the timestamp
            formatted = ts + output.replace('\n', '\n' + ts)
            # Remove trailing empty prefix if output ended in newline
            if formatted.endswith(ts):
                formatted = formatted[:-len(ts)]
                
            with open(self.log_file, 'a') as f:
                f.write(formatted)
                
        return output

    def load_map(self, map_name, wait_for_load=True):
        print(f"Loading map '{map_name}'...")
        self.send_command(f"map {map_name}")
        
        if wait_for_load:
            print("Waiting for map to finish loading (waiting for player to spawn)...")
            
            # Step 1: Wait for the engine to acknowledge the map load
            # This ensures we don't accidentally read properties from the main menu background map
            map_started = False
            start_time = time.time()
            while time.time() - start_time < 15:
                out = self.read_console(print_to_terminal=False)
                if "Host_NewGame" in out:
                    map_started = True
                    break
                time.sleep(0.5)
                
            if not map_started:
                print("Warning: Engine did not acknowledge map load start.")
            
            # Step 2: Poll 'getpos' until it responds with 'setpos'
            # Now that the transition started, getpos will only succeed when the new map is loaded and player spawns.
            start_time = time.time()
            while time.time() - start_time < 45: # 45 seconds max for a large map
                self.send_command("getpos")
                time.sleep(1.0)
                out = self.read_console(print_to_terminal=False)
                if "setpos" in out:
                    print("Map loaded successfully and player has spawned!")
                    # Sleep slightly to let the engine settle after spawning
                    time.sleep(1.0)
                    return True
            
            print("Warning: Timed out waiting for map to load.")
            return False

    def move_forward(self, duration=None, state=True):
        if duration is not None:
            print(f"Walking forward for {duration}s...")
            self.send_command("+forward")
            time.sleep(duration)
            self.send_command("-forward")
        else:
            cmd = "+forward" if state else "-forward"
            self.send_command(cmd)

    def move_backward(self, duration=None, state=True):
        if duration is not None:
            print(f"Walking backward for {duration}s...")
            self.send_command("+back")
            time.sleep(duration)
            self.send_command("-back")
        else:
            cmd = "+back" if state else "-back"
            self.send_command(cmd)

    def move_left(self, duration=None, state=True):
        if duration is not None:
            print(f"Walking left for {duration}s...")
            self.send_command("+moveleft")
            time.sleep(duration)
            self.send_command("-moveleft")
        else:
            cmd = "+moveleft" if state else "-moveleft"
            self.send_command(cmd)

    def move_right(self, duration=None, state=True):
        if duration is not None:
            print(f"Walking right for {duration}s...")
            self.send_command("+moveright")
            time.sleep(duration)
            self.send_command("-moveright")
        else:
            cmd = "+moveright" if state else "-moveright"
            self.send_command(cmd)
        
    def jump(self):
        print("Jumping...")
        self.send_command("+jump")
        time.sleep(0.1)
        self.send_command("-jump")
        
    def interact(self):
        print("Interacting (use)...")
        self.send_command("+use")
        time.sleep(0.1)
        self.send_command("-use")

    def enable_io_logging(self):
        """Enables Source Engine developer logging so entity I/O (like brush triggers) print to the console."""
        print("Enabling Engine I/O logging...")
        self.send_command("developer 1")
        self.send_command("ent_messages_draw 1")
        
    def check_event(self, event_substring, clear_buffer=True):
        """
        Checks if a specific string (like a brush trigger name) was logged in the console.
        Returns True if found, False otherwise.
        """
        # Ensure we have the latest console output
        self.read_console(print_to_terminal=False)
        
        found = False
        for line in self.event_buffer:
            if event_substring in line:
                found = True
                break
                
        if clear_buffer:
            self.event_buffer.clear()
            
        return found

    def init_virtual_mouse(self):
        """Initializes a virtual mouse device using evdev for hardware-level simulation."""
        try:
            from evdev import UInput, ecodes as e
            capabilities = {
                e.EV_KEY: [e.BTN_LEFT, e.BTN_RIGHT],
                e.EV_REL: [e.REL_X, e.REL_Y],
            }
            self.ui = UInput(capabilities, name="portal2-wrapper-mouse")
            return True
        except ImportError:
            print("Warning: 'evdev' module not found. Mouse simulation disabled.")
            return False
        except Exception as ex:
            print(f"Warning: Could not create virtual mouse ({ex}). You may need to run with sudo.")
            return False

    def move_mouse(self, dx, dy, steps=1, delay=0.01):
        """Simulates raw mouse movement using evdev."""
        print(f"Moving mouse (dx={dx}, dy={dy}, steps={steps})...")
        if not self.ui:
            if not self.init_virtual_mouse():
                return
                
        try:
            from evdev import ecodes as e
            for _ in range(steps):
                if dx != 0:
                    self.ui.write(e.EV_REL, e.REL_X, dx)
                if dy != 0:
                    self.ui.write(e.EV_REL, e.REL_Y, dy)
                self.ui.syn()
                time.sleep(delay)
        except Exception as ex:
            print(f"Error during mouse movement: {ex}")

    def fire_left(self, state=True):
        cmd = "+attack" if state else "-attack"
        self.send_command(cmd)

    def fire_right(self, state=True):
        cmd = "+attack2" if state else "-attack2"
        self.send_command(cmd)

    def jump_state(self, state=True):
        cmd = "+jump" if state else "-jump"
        self.send_command(cmd)
        
    def crouch_state(self, state=True):
        cmd = "+duck" if state else "-duck"
        self.send_command(cmd)
        
    def interact_state(self, state=True):
        cmd = "+use" if state else "-use"
        self.send_command(cmd)

    def play_csv(self, csv_path, fps=20, mouse_scale=1.0, countdown=3):
        import csv
        import time
        if countdown > 0:
            print(f"\n--- Testing Playback ---")
            print(f"Hands off the keyboard! Playing back in {countdown} seconds...")
            time.sleep(countdown)
        print(f"Playing back {csv_path} at {fps} FPS (Mouse scale: {mouse_scale})...")
        tick_duration = 1.0 / fps
        
        state = {
            'w': 0, 'a': 0, 's': 0, 'd': 0,
            'jump': 0, 'crouch': 0, 'use': 0, 'fire_left': 0, 'fire_right': 0
        }
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            next_tick = time.time()
            for row in reader:
                dx = int(int(row['mouse_dx']) * mouse_scale)
                dy = int(int(row['mouse_dy']) * mouse_scale)
                
                if int(row['move_w']) != state['w']:
                    state['w'] = int(row['move_w'])
                    self.move_forward(state=bool(state['w']))
                if int(row['move_a']) != state['a']:
                    state['a'] = int(row['move_a'])
                    self.move_left(state=bool(state['a']))
                if int(row['move_s']) != state['s']:
                    state['s'] = int(row['move_s'])
                    self.move_backward(state=bool(state['s']))
                if int(row['move_d']) != state['d']:
                    state['d'] = int(row['move_d'])
                    self.move_right(state=bool(state['d']))
                if int(row['jump']) != state['jump']:
                    state['jump'] = int(row['jump'])
                    self.jump_state(state=bool(state['jump']))
                if int(row['crouch']) != state['crouch']:
                    state['crouch'] = int(row['crouch'])
                    self.crouch_state(state=bool(state['crouch']))
                if int(row['use']) != state['use']:
                    state['use'] = int(row['use'])
                    self.interact_state(state=bool(state['use']))
                if int(row['fire_left']) != state['fire_left']:
                    state['fire_left'] = int(row['fire_left'])
                    self.fire_left(state=bool(state['fire_left']))
                if int(row['fire_right']) != state['fire_right']:
                    state['fire_right'] = int(row['fire_right'])
                    self.fire_right(state=bool(state['fire_right']))
                
                # Smoothly interpolate mouse movement over the 50ms tick to prevent acceleration spikes
                mouse_steps = 10
                step_sleep = tick_duration / mouse_steps
                step_dx = dx / mouse_steps
                step_dy = dy / mouse_steps
                acc_dx = 0.0
                acc_dy = 0.0
                
                for _ in range(mouse_steps):
                    loop_start = time.time()
                    
                    acc_dx += step_dx
                    acc_dy += step_dy
                    send_dx = int(acc_dx)
                    send_dy = int(acc_dy)
                    acc_dx -= send_dx
                    acc_dy -= send_dy
                    
                    if send_dx != 0 or send_dy != 0:
                        # Call internal ui directly to avoid console spam from move_mouse
                        if not self.ui:
                            self.init_virtual_mouse()
                        if self.ui:
                            from evdev import ecodes as e
                            if send_dx != 0: self.ui.write(e.EV_REL, e.REL_X, send_dx)
                            if send_dy != 0: self.ui.write(e.EV_REL, e.REL_Y, send_dy)
                            self.ui.syn()
                            
                    elapsed = time.time() - loop_start
                    if step_sleep - elapsed > 0:
                        time.sleep(step_sleep - elapsed)
                
                next_tick += tick_duration
                sleep_time = next_tick - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        # Reset all keys
        self.move_forward(state=False)
        self.move_left(state=False)
        self.move_backward(state=False)
        self.move_right(state=False)
        self.jump_state(state=False)
        self.crouch_state(state=False)
        self.interact_state(state=False)
        self.fire_left(state=False)
        self.fire_right(state=False)
        print("Playback finished.")

    def start_recording(self, fps=20, countdown=3, duration=None, outcome="manual_test"):
        import time
        if countdown > 0:
            print(f"\n--- Testing Programmatic Recording ---")
            print(f"Recording will start in {countdown} seconds. Move around!")
            time.sleep(countdown)
            
        if not self.recorder:
            from recorder import EpisodeRecorder
            self.recorder = EpisodeRecorder(fps=fps, controller=self)
            self.recorder.tracker.start()
            self.recorder.camera.start()
        self.recorder.start_recording()
        
        if duration is not None:
            print(f"Recording for {duration} seconds...")
            time.sleep(duration)
            self.stop_recording(outcome=outcome)

    def stop_recording(self, outcome="unknown"):
        if self.recorder:
            self.recorder.stop_recording(outcome=outcome)

    def disconnect(self):
        if self.sock:
            self.sock.close()
            self.sock = None
        if self.ui:
            self.ui.close()
            self.ui = None


def is_game_running():
    """Checks if the Portal 2 process is running."""
    try:
        # Looking for common Portal 2 process names
        subprocess.check_output(["pgrep", "-f", "portal2_linux|hl2_linux|portal2|portal2.exe"])
        return True
    except subprocess.CalledProcessError:
        return False


def launch_game(port=8020):
    """Launches Portal 2 via Steam with the netconport parameter."""
    print("Game not detected. Launching Portal 2...")
    # App ID 620 is Portal 2
    subprocess.Popen(["steam", "-applaunch", "620", "-netconport", str(port)])
    
    print("Waiting for game to start...")
    for i in range(60):
        if is_game_running():
            print("Process detected! Giving the engine time to initialize the network port...")
            time.sleep(10) # Give the engine extra time to open the port after process starts
            return True
        time.sleep(1)
    
    print("Timeout waiting for game to launch.")
    return False


def main():
    port = 8020
    
    if not is_game_running():
        if not launch_game(port):
            return
    else:
        print("Portal 2 is already running.")

    controller = Portal2Controller(port)
    
    # Try connecting multiple times in case the game is still loading
    connected = False
    for attempt in range(5):
        if controller.connect():
            connected = True
            break
        print(f"Retrying connection ({attempt + 1}/5)...")
        time.sleep(3)
        
    if not connected:
        print("Could not connect to the game console. Make sure it launched with -netconport 8020")
        return

    # Clear initial connection spam
    time.sleep(0.5)
    controller.read_console(print_to_terminal=False)

    print("\n--- Starting Test Procedure ---")
    
    # Load a standard test map (avoids the 30-second unskippable bed intro of chapter 1)
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
                
    print(f"Baseline established. Attempting auto-jump check...")
    
    # Poll until the jump successfully changes the position
    control_verified = False
    while not control_verified:
        # Try to jump
        controller.jump()
        # Give the physics engine a moment to apply vertical velocity
        time.sleep(0.3) 
        
        controller.send_command("getpos")
        time.sleep(0.2)
        out = controller.read_console(print_to_terminal=False)
        
        current_pos = baseline_pos
        # Read lines in reverse to grab the most recent getpos output
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
                    
    print("\nJump successful! Player has active control.")
    print("Wrapper initialization complete. Ready for external scripts to send commands.")
    
    controller.disconnect()


if __name__ == "__main__":
    main()
