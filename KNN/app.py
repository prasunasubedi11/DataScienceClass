from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load('knn_model.pkl')
scaler = joblib.load('scaler.pkl')

class Student(BaseModel):
    age: int
    study_hours: float
    attendance: float

@app.get("/")
def home():
    return{"message": "KNN Students Result Prediction API"}


@app.post("/predict")
def predict(student: Student):

    data = [[
        student.age,
        student.study_hours,
        student.attendance
    ]]

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)[0]

    return{
        "prediction":prediction
    }