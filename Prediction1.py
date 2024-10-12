from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load the logistic regression model and label encoder
try:
    with open('best_model.pkl', 'rb') as model_file:
        model_logreg = joblib.load(model_file)
    scaler_logreg = joblib.load('logreg_scaler.pkl')
    onehot_encoder = joblib.load('onehot_encoder.pkl')
    ordinal_encoder = joblib.load('ordinal_encoder.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    logger.info("Model and encoders loaded successfully")
except Exception as e:
    logger.error(f"Error loading models and encoders: {e}")
    raise HTTPException(status_code=500, detail="Error loading models and encoders")

class DataInput(BaseModel):
    age: int
    job: str  
    marital: str  
    education: str  
    default: str  
    housing: str  
    loan: str  
    contact: str  
    month: str  
    day_of_week: str  
    duration: float
    campaign: int
    pdays: int
    previous: int
    poutcome: str  

def preprocess_data(data: dict):
    data_df = pd.DataFrame([data])

    nominal_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']
    ordinal_columns = ['month', 'day_of_week']
    numerical_columns = ['age', 'duration', 'campaign', 'pdays', 'previous']

    # Handle unknown categories for nominal columns
    for col in nominal_columns:
        known_categories = set(onehot_encoder.categories_[nominal_columns.index(col)])
        data_df[col] = data_df[col].apply(lambda x: x if x in known_categories else 'unknown')

    # Handle unknown categories for ordinal columns
    for col in ordinal_columns:
        known_categories = set(ordinal_encoder.categories_[ordinal_columns.index(col)])
        data_df[col] = data_df[col].apply(lambda x: x if x in known_categories else 'unknown')

    # One-Hot Encoding for nominal columns
    one_hot_encoded_data = onehot_encoder.transform(data_df[nominal_columns])
    one_hot_encoded_df = pd.DataFrame(one_hot_encoded_data, columns=onehot_encoder.get_feature_names_out(nominal_columns), index=data_df.index)

    # Ordinal Encoding for ordinal columns
    ordinal_encoded_data = ordinal_encoder.transform(data_df[ordinal_columns])
    ordinal_encoded_df = pd.DataFrame(ordinal_encoded_data, columns=ordinal_columns, index=data_df.index)

    # Concatenate all data frames
    processed_data = pd.concat([data_df[numerical_columns], one_hot_encoded_df, ordinal_encoded_df], axis=1)

    return processed_data

@app.post("/predict/")
def predict(data: DataInput):
    try:
        data_dict = data.dict()
        preprocessed_data = preprocess_data(data_dict)
        scaled_data = scaler_logreg.transform(preprocessed_data)
        prediction = model_logreg.predict(scaled_data)

        # Assuming prediction is a single value array
        predicted_value = label_encoder.inverse_transform(prediction)[0]

        return {"prediction": str(predicted_value)}  # Ensure JSON serializability
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")

@app.get("/")
def read_root():
    return {"message": "Hello World"}
