import os


START_SEQ="\033[90m"
END_SEQ="\033[0m"

def log(text):
    if(os.getenv("DEBUG") == "True"):
        print(f"   {START_SEQ}{text}{END_SEQ}")