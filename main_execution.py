import subprocess

def run_script(script_path):
    try:
        result = subprocess.run(['python', script_path], check=True, capture_output=True, text=True)
        print(f"Execution successful: {script_path}")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while executing {script_path}: {e}")
        print(e.stderr)

if __name__ == "__main__":
    scripts = [
        "C:/ai_integration/Face_Recogniton/dataNface.py",
        "C:/ai_integration/Person_Detection/realtime_helmet_detection.py",
        "C:/ai_integration/Person_Detection/realtime_person_detection.py",
        "C:/ai_integration/Crosswalk_Detection/crosswalk_detection.py"
    ]
    
    for script in scripts:
        run_script(script)
