# Sensor Placement — Quantum Application

This folder contains the standalone quantum sensor-placement Streamlit application and its supplied supporting Python modules.

## Run locally

```bash
pip install -r requirements.txt
python -m streamlit run quantum_app.py
```

Streamlit will print the local URL, usually `http://localhost:8501`.

## Deploy with Streamlit Community Cloud

Upload the contents of this folder to a GitHub repository, then select `quantum_app.py` as the app's main file.

## Important

The uploaded ZIP did not include the original `requirements.txt` or any other project files beyond the Python files listed here. Review imports and add any additional data/assets/modules required by your own app before deployment. This app's local simulator is not a physical quantum computer.
