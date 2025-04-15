import os
import threading
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__)).replace("src\\Utils", "")
LUA_DIR = os.path.join(BASE_DIR, "Lua")
CURRICULUM_DIR = os.path.join(BASE_DIR, "training_examples", "out")

def list_and_process_files(directory, file_list):
    """Function to list all files and process them with Lua."""
    try:
        for entry in os.scandir(directory):
            if entry.is_file():
                file_list.append(entry.path)
                process_lua_file(entry.path)
            elif entry.is_dir():
                thread = threading.Thread(target=list_and_process_files, args=(entry.path, file_list))
                thread.start()
                thread.join()
    except PermissionError:
        pass  # Ignore directories we don't have permission to access

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

def main(directory):
    """Main function to initiate file listing and processing."""
    file_list = []
    main_thread = threading.Thread(target=list_and_process_files, args=(directory, file_list))
    main_thread.start()
    main_thread.join()
    
    for file in file_list:
        print(f"Processed: {file}")

        
if __name__ == "__main__":
    main(os.path.join(LUA_DIR, "Curriculum1"))
