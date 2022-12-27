from http import HTTPStatus
from fastapi import FastAPI, Request
from dataclasses import dataclass
import numpy as np
import uvicorn
import pickle

# import models

pickle_knn_pip =  open('/Users/aleenkmail/Desktop/CarPricePrediction2/models/knn_pipline.pkl', 'rb')
pickle_gbt_pip =  open('/Users/aleenkmail/Desktop/CarPricePrediction2/models/gbt_pipline.pkl', 'rb')
knn_model = pickle.load(pickle_knn_pip)
gbt_model = pickle.load(pickle_gbt_pip)



# Define application
app = FastAPI(
    title="Car Price Prediction EndPoint",
    version="0.1",
    )


# define car data class with all used features
@dataclass
class Car:
    year : int
    engine : int
    meter : int
    prev_owners : int
    passengers : int
    name : str
    color : str
    fueil : str
    prev_use : str
    license : str
    gear : str
    glass : str 
    weel_drive : str


"""Health check."""
@app.get("/")
def _index(request: Request):
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {},
    }
    return response


"""endpoint to predict the price"""
@app.post("/predict")
def _get_predict(feature: Car):

    
    color = feature.color
    glass = feature.glass
    name = feature.name
    fueil = feature.fueil
    prevـuse = feature.prev_use
    license = feature.license
    gear = feature.gear
    weel_drive = feature.weel_drive
    year = feature.year
    engine = feature.engine
    passengers = feature.passengers
    meter = feature.meter
    prev_owners = feature.prev_owners
    
    # create list with values of features
    data =  np.array([color, fueil, license, name, gear, glass, weel_drive,\
         prevـuse, prev_owners, passengers, engine, meter, year]).reshape(1, -1)
    


    print(data)

    # predict the price
    knn_predicted_price = knn_model.predict(data)
    gbt_predicted_price = gbt_model.predict(data)
 
    
    return {"price predicted with knn :" : int(knn_predicted_price[0]),
            "price predicted with gbt :" : int(gbt_predicted_price[0]),
            }

if __name__ == "__main__":
    uvicorn.run("service:app", host= '0.0.0.0', port= 8000, reload= True)



