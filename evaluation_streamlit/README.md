# Streamlit App for Model Evaluation

## Description

This is a simple example of a Streamlit app that evaluates the results of different model predictions in a classic binary classification problem. Given an **EVALUATION CSV** file, the app displays the false negative patients on a web UI, based on some input threshold and desired specificity.

## Setup

It is recommended to use the Python version specified in the `.python-version` file. However, any compatible version should work. We also recommend using a virtual environment (venv) to manage dependencies and avoid conflicts.

### Setup Instructions

1. **Create a virtual environment (venv)**  
   In your terminal, navigate to the project directory and run the following commands to set up a virtual environment:

   ```bash
   # Create a virtual environment
   python -m venv myProjectVenv

   # Activate the virtual environment
   source myProjectVenv/bin/activate  # On Linux/MacOS
   # For Windows: 
   myProjectVenv\Scripts\activate
   # For Windows WLS: 
   source myProjectVenv/Scripts/Activate

2. **Install Dependencies**
    After activating the virtual environment, install the required dependencies listed in the `requirements/base.txt` file by running:
    ```bash
    # Install dependencies
    pip install -r requirements/base.txt

3. **Run the Streamlit app**
    To start the Streamlit app, make sure your virtual environment is activated and run the following command to get Local URL where the UI is exposed:
    ```bash
    strearmlit run evaluation/evaluation_app.py # ctrl+C to stop it

4. **Deactivate you Virtual Env**
    After you worked with and you do not need to have activated you Venv you can deactivate it by:
    ```bash
    deactivate

## Steamlit Evaluation App Usage
The web app UI is gonna be available at `Local URL: http://localhost:8501`, upload your `predictionsEvaluation.csv` and insert some thresholds to test for example [0.4, 0.7, 0.8] and the desired specificity like 0.8
