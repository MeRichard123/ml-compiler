import subprocess
import os
from tqdm import tqdm

BASELINE_DIR = os.path.dirname(os.path.abspath(__file__))
QWEN_DIR = os.path.join(BASELINE_DIR, "Starcoder2")
BASE_DIR = os.path.dirname(os.path.abspath(__file__)).replace("Baseline", "")
TRAINING_EXAMPLES_DIR = os.path.join(BASE_DIR, "training_examples", "All")
files = os.listdir(TRAINING_EXAMPLES_DIR)
current_files = os.listdir(QWEN_DIR)

model = "starcoder2:latest"


for file in tqdm(files):
    program = ""
    prompt = "Here is some lua code, please give me the output and only the out of the program:\n\n"
    if file.endswith(".luax") and file not in current_files:
        with open(os.path.join(TRAINING_EXAMPLES_DIR, file), "r") as f:
            try:
                content = f.read().split("<PROGRAM END>")[0]
            except:
                print(f"Error reading file {file}. Skipping...")
                continue
            
            prompt += content + "\n\n"
            program = content

        res = subprocess.run(
            ["ollama", "run", model, prompt]
            , capture_output=True, text=True
        ).stdout.replace("```", "")

        with open(os.path.join(QWEN_DIR, file), "w") as f:
            f.write(program)
            f.write("<PROGRAM END>\n")
            f.write(res)


