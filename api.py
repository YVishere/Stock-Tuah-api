from fastapi import FastAPI
import glob
from fastapi.middleware.cors import CORSMiddleware
import os
import genNext as gn

import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

temp_csv_files = glob.glob('datasets_chosen/*.csv')
csv_files = [os.path.basename(file) for file in temp_csv_files]

modded_csv_filesTemp = glob.glob('modded_datasets/*.csv')
modded_csv_files = [os.path.basename(file) for file in modded_csv_filesTemp]

@app.get("/")
def main_page():
    return "Hello World"

@app.get("/listdatasets")
def list_csv_files():
    if len(modded_csv_files) == 0:
        return csv_files
    return modded_csv_files

@app.get("/dayinc")
def dayincrement(x : int):
    return gn.main(x)

for endpoint in csv_files:
    @app.get(f"/{endpoint}")
    def dynamic_endpoint(endpoint = endpoint):
        if len(modded_csv_files) == 0:
            data = pd.read_csv(f"datasets_chosen/{endpoint}")
        else: data = pd.read_csv(f"modded_datasets/{endpoint}")
        return {"Date": data['Date'].values.tolist(), "Price": data['Close'].values.tolist(), "Volume": data['Volume'].values.tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)