from http import HTTPStatus
from fastapi import FastAPI, Request
from dataclasses import dataclass
import numpy as np
import uvicorn
import pickle

# import pipline
pickle_pip =  open('/Users/aleenkmail/Desktop/CarPricePrediction2/pip_model.pkl', 'rb') 
pipline = pickle.load(pickle_pip)


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
    
    # features = np.array([[color, glass ,name, fueil, prevـuse, license, gear, weel_drive, year, engine, passengers, meter, prev_owners]]).reshape(1, -1)
    # # data= pd.DataFrame(c, columns=['year', 'engine', 'meter','prev_owners', 'passengers', 'name',\
    # #      'color', 'fueil', 'prevـuse', 'license', 'gear', 'glass', 'weel_drive'])
    # # features = np.array(list(vars(feature).values())[0:-1])


    print(data)

    # predict the price
    predicted_price = pipline.predict(data)

 
    
    return {"predicted" : int(predicted_price[0]) }

if __name__ == "__main__":
    uvicorn.run("service:app", host= '0.0.0.0', port= 8000, reload= True)



