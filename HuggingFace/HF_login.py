import os
import subprocess

from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("HF_TOKEN")

subprocess.run(["huggingface-cli", "login", "--token", TOKEN])
print("LOGIN HUGGINGFACE SUCCESSFULLY!")