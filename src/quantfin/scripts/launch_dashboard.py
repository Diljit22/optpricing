import os
import sys
from streamlit.web import cli as stcli

def main():
    # Path to the dashboard script inside the package
    here = os.path.dirname(__file__)
    dashboard_script = os.path.join(here, "dashboard.py")

    # Simulate: streamlit run dashboard_script
    sys.argv = ["streamlit", "run", dashboard_script]
    stcli.main()
