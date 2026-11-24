
import ast
import itertools
import subprocess
import sys
import time

# Get command-line arguments
#REAL SPC,TPC,UC,SBX,NONE  MPOLY,MNUF,NONE False True
arguments = sys.argv[1:]

encoding = arguments[0]
cross_list = arguments[1].split(',')
mutation_list = arguments[2].split(',')
int2 = arguments[3]
hyper_mut = arguments[4]
# Generate all combinations
combinations = itertools.product(cross_list, mutation_list)
# Create one tmux session per combination
for crossover, mutation in combinations:
    session_name = f"{encoding}_{crossover}_{mutation}_{'INT2' if int2 == 'True' else 'INT1'}_{'M' if hyper_mut == 'True' else ''}"
    # Check whether the tmux session already exists
    session_exists = subprocess.run(  ["tmux", "has-session", "-t", session_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    # Kill the existing session if it exists
    if session_exists:
        print(f"Deleting existing session: {session_name}")
        subprocess.run( ["tmux", "kill-session", "-t", session_name], check=True)
    # Build the Python command
    python_command = ["python3", "-u", "main.py",  encoding,  crossover,  mutation,  int2, hyper_mut]
    # Create the shell command
    shell_command = (subprocess.list2cmdline(python_command) if sys.platform == "win32"  else " ".join(subprocess.list2cmdline([arg]).replace('"', '\\"')  for arg in python_command))
    # Keep the terminal open after Python finishes
    shell_command += "; exec bash"
    # Create a new tmux session
    print(f"Creating session: {session_name}")
    subprocess.run([ "tmux", "new-session", "-d", "-s", session_name, "bash", "-c", shell_command], check=True)
    print(f"Session created successfully: {session_name}")
    time.sleep(0.1)

print("\nAll tmux sessions have been created.")
