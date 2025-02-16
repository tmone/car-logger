import os
import sys

# Add src directory to Python path
src_dir = os.path.join(os.path.dirname(__file__), 'src')
sys.path.append(src_dir)

# Run Streamlit app
os.system(f"streamlit run {os.path.join(src_dir, 'dashboard', 'app.py')}")
