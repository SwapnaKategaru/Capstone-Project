
import json
import numpy as np
import pandas as pd
import os
import joblib
import pickle

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType

input_sample = pd.DataFrame({"age": pd.Series([0.0], dtype="float64"), "anaemia": pd.Series([0], dtype="int64"), "creatinine_phosphokinase": pd.Series([0.0], dtype="float64"), "diabetes": pd.Series([0], dtype="int64"), "ejection_fraction": pd.Series([0.0], dtype="float64"), "high_blood_pressure": pd.Series([0], dtype="int64"), "platelets": pd.Series([0.0], dtype="float64"), "serum_creatinine": pd.Series([0.0], dtype="float64"), "serum_sodium": pd.Series([0.0], dtype="float64"), "sex": pd.Series([0], dtype="int64"), "smoking": pd.Series([0], dtype="int64"), "time": pd.Series([0.0], dtype="float64")})
output_sample = np.array([0])


def init():
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'automlmodel.pkl')
    print("Found model:", os.path.isfile(model_path)) 
    model = joblib.load(model_path)

@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))

def run(data):
    try:
        result = model.predict(data)
        return result.tolist()
    except Exception as e:
        result = str(e)
        return result  
