import socket
import time

HOST = '127.0.0.1'
PORT = 8020

def test_portal2_io():
    print(f"Connecting to Portal 2 on {HOST}:{PORT}...")

    try:
        # Create a raw TCP socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(3.0) 
            s.connect((HOST, PORT))
            print("Socket connected.\n")

            # Source engine requires \r\n line endings
            test_string = "hello_from_python"
            command = f"echo {test_string}\r\n"
            
            print(f"Sending: {command.strip()}")
            s.sendall(command.encode('ascii'))

            # Give the game engine a fraction of a second to process and reply
            time.sleep(0.1)

            # Read up to 4KB of the response buffer
            response = s.recv(4096).decode('ascii', errors='ignore')

            print("--- Output Buffer ---")
            print(response.strip())
            print("---------------------")

            if test_string in response:
                print("\nSUCCESS: Raw I/O verified. Portal 2 received and executed the command.")
            else:
                print("\nWARNING: Connected, but the echo string wasn't found in the recent console output.")

    except ConnectionRefusedError:
        print("\nERROR: Connection refused. Is Portal 2 currently running with '-netconport 8020'?")
    except Exception as e:
        print(f"\nERROR: {e}")

if __name__ == "__main__":
    test_portal2_io()