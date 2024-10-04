from flask import Flask,request,render_template
from src.pipelines.prediction_pipeline import *
import numpy as np
import pandas as pd

application = Flask(__name__)
app = application

@app.route("/")
def index():
    return render_template("home.html")

@app.route("/predict",methods=["GET","POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    else:
        data = CustomData(
            Income = request.form.get("Income"),
            Age = request.form.get("Age"),
            CURRENT_HOUSE_YRS = request.form.get("CURRENT_HOUSE_YRS"),
            Experience = request.form.get("Experience"),
            House_Ownership = request.form.get("House_Ownership"),
            Car_Ownership = request.form.get("Car_Ownership"),
            Married_Single = request.form.get("Married_Single")
        ).data_as_data_frame()
        
        result = PredictionPipeline().predict(data)[0]
        return render_template("home.html",result=result)
    
    
if __name__ == "__main__":
    app.run(host="0.0.0.0")
        
        