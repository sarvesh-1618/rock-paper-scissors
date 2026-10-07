from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import pickle
import pandas as pd
import uvicorn


# Create FastAPI application
app = FastAPI()


# Template folder
templates = Jinja2Templates(directory="templates")


# Static folder
app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


# Load trained model
with open("california_model.pkl", "rb") as f:
    classifier = pickle.load(f)


# Features used by the trained model
required_columns = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "Population",
    "AveOccup",
    "Latitude"
]


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def main_page(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request}
    )


# --------------------------------------------------
# MANUAL PREDICTION
# --------------------------------------------------

@app.get("/predict")
def predict(
    MedInc: float,
    HouseAge: float,
    AveRooms: float,
    Population: float,
    AveOccup: float,
    Latitude: float
):

    # Create input DataFrame
    input_data = pd.DataFrame(
        [[
            MedInc,
            HouseAge,
            AveRooms,
            Population,
            AveOccup,
            Latitude
        ]],
        columns=required_columns
    )

    # Make prediction
    prediction = classifier["model"].predict(input_data)

    return {
        "prediction": float(prediction[0])
    }


# --------------------------------------------------
# CSV PREDICTION
# --------------------------------------------------

@app.post("/predict_file")
def predict_file(file: UploadFile = File(...)):

    # Read CSV file
    df = pd.read_csv(file.file)


    # Check missing columns
    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]


    if missing_columns:

        return {
            "error": "Missing required columns",
            "missing_columns": missing_columns
        }


    # Select only columns required by model
    input_data = df[required_columns]


    # Make predictions
    predictions = classifier["model"].predict(input_data)


    # Return predictions
    return {
        "rows": len(predictions),
        "predictions": predictions.tolist()
    }


# --------------------------------------------------
# RUN SERVER
# --------------------------------------------------

if __name__ == "__main__":

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )