import subprocess
from quantfin.config import PROJECT_ROOT # Use our centralized config for paths

def launch_dashboard():
    """
    This function is the entry point for the 'quantfin-dashboard' console script.
    It finds the main app.py file and launches it using Streamlit.
    """
    # Construct the full path to the main app.py file in the scripts directory
    app_path = PROJECT_ROOT / "scripts" / "app.py"
    
    print(f"Attempting to launch Streamlit app from: {app_path}")
    
    if not app_path.exists():
        print(f"\nError: Dashboard entry point not found at '{app_path}'.")
        print("Please ensure 'scripts/app.py' exists in your project root.")
        return

    try:
        # Use subprocess to execute the command: streamlit run scripts/app.py
        subprocess.run(["streamlit", "run", str(app_path)], check=True)
    except FileNotFoundError:
        print("\nError: 'streamlit' command not found.")
        print("Please ensure Streamlit is installed in your environment (`pip install quantfin[app]`).")
    except subprocess.CalledProcessError as e:
        print(f"\nAn error occurred while running the Streamlit app: {e}")

def run_benchmark():
    """Entry point for running the full benchmark."""
    benchmark_script = PROJECT_ROOT / "scripts" / "run_full_benchmark.py"
    print(f"Executing benchmark script: {benchmark_script}")
    subprocess.run(["python", str(benchmark_script)], check=True)