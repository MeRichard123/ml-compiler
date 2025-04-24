import os
import threading
import subprocess
from concurrent.futures import ThreadPoolExecutor

BASE_DIR = os.path.dirname(os.path.abspath(__file__)).replace("src\\Utils", "")
LUA_DIR = os.path.join(BASE_DIR, "Lua")
CURRICULUM_DIR = os.path.join(BASE_DIR, "training_examples", "out")

def process_lua_file(file_path):
    """Runs a Lua file and writes output to a .luax file."""
    try:
        result = subprocess.run(["lua", file_path], capture_output=True, text=True)
        output = result.stdout.strip()
        
        with open(file_path, "r") as original_file:
            content = original_file.read()
        
        # edit this to save inside of training_examples/Curriculum1
        filename = file_path.split("\\")[-1] + "x"
        new_file_path = os.path.join(CURRICULUM_DIR, filename)

        with open(new_file_path, "w") as new_file:
            new_file.write(content + "\n<PROGRAM END>\n" + output)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def list_and_process_files(directory, file_list, executor, lock):
    """Thread-safe parallel file/directory processing."""
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_file():
                    with lock:  # Thread-safe append
                        file_list.append(entry.path)
                    executor.submit(process_lua_file, entry.path)
                elif entry.is_dir():
                    executor.submit(list_and_process_files, entry.path, file_list, executor, lock)
    except PermissionError:
        pass

def main(directory):
    """Main function with controlled parallelism."""
    file_list = []
    lock = threading.Lock()
    
    # Use ThreadPoolExecutor (limits max threads)
    with ThreadPoolExecutor(max_workers=8) as executor:
        list_and_process_files(directory, file_list, executor, lock)
    
    print(f"Processed {len(file_list)} files.")

if __name__ == "__main__":
   main(os.path.join(LUA_DIR, "Curriculum1"))
