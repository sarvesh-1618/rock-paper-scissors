


python -m uvicorn app1:app --host 0.0.0.0 --port 8000 --reload

--run this to get output >>>>.



what are the changes to do in fastapi ???(diff in flask and fastapi)
========================================================

change-1:    from flask import Flask,uvicorn
             from fastapi import FastAPI

change-2:    app=Flask(__name__)
             app=FastAPI()

change-3:    @app.route('predict',methods=['post'])
             @app.post('predict')
             arguments you can provide directly in function call

change-4:    app.run()
             uvicorn.run(app,host,port)

change-5:    python filename.py
             python -m uvicorn filename:app --port --host

change-6:    http:0.0.0.0:5000  ==== it will not work
             http:127.0.0.1:5000  run


             ========================================
ChatGPT:::(app.py codebefore)
Hi Your role is good UI developer
I have developed a fastapi  application i want to add UI layer on that for frontend 
add HTML CSS style make a clear UI

I also provided my app code

from fastapi import FastAPI, UploadFile, File
import pickle
import pandas as pd
import uvicorn  # It is a server to run FastAPI

# Initialize app
app = FastAPI()   

# Load model
with open("california_model.pkl", "rb") as f:
    classifier = pickle.load(f)

@app.get("/")   
def main_page():
    return('welcome')
# Predict from query parameters

@app.get("/predict")
def predict(MedInc:float,HouseAge:float,AveRooms:float, 
           Population:float, AveOccup:float, Latitude:float):
    input_data = [[MedInc,HouseAge,AveRooms, 
           Population, AveOccup, Latitude]]
    prediction = classifier['model'].predict(input_data)
    return {"prediction": float(prediction[0])}

# Predict from uploaded CSV file
@app.post("/predict_file")
def predict_file(file: UploadFile = File(...)):
    df_test = pd.read_csv(file.file)
    prediction = classifier['model'].predict(df_test)
    return {"predictions": prediction.tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)



==================================================================


(install packages)
===================
pip install jinja2
pip install -r requirements.txt


**If jinja2 isn't in requirements.txt, add:**
::::---jinja2