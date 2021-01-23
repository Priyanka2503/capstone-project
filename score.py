import json
import numpy as np
import os
import joblib
import pandas as pd

def init():
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    print("Found model:", os.path.isfile(model_path)) #To check whether the model is actually present on the location we are looking at
    model = joblib.load(model_path)
def run(data):
    try:
        data = json.loads(data)['data']
        data = pd.DataFrame.from_dict(data)
        result = model.predict(data)
    except Exception as e:
        error = str(e)
        return error