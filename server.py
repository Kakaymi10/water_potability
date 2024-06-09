import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import numpy as np
import multiprocessing as mp

app = FastAPI()

# Load the TensorFlow model
model = tf.keras.models.load_model('model_l1.keras')

# Load the scaler and fit it to your data
scaler = StandardScaler()
# Assuming you have your data stored in a numpy array called 'X'
# You may load your dataset here and fit the scaler
# scaler.fit(X)

class InputData(BaseModel):
    data: List[List[float]]

@app.get("/greet/{name}")
async def greet_user(name: str):
    return {"message": f"Hello, {name}!"}

@app.post("/predict/")
async def predict(data: InputData):
    # Convert data to numpy array
    input_data = np.array(data.data)
    
    # Transform the input data using the fitted scaler
    input_data_scaled = scaler.transform(input_data)
    
    # Make predictions
    predictions = model.predict(input_data_scaled)
    
    return {"predictions": predictions.tolist()}
