from fastapi import FastAPI
import glob
from fastapi.middleware.cors import CORSMiddleware
import os

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

@app.get("/")
def main_page():
    return "Hello World"

@app.get("/listdatasets")
def list_csv_files():
    return csv_files

for endpoint in csv_files:
    @app.get(f"/{endpoint}")
    def dynamic_endpoint(endpoint = endpoint):
        data = pd.read_csv(f"datasets_chosen/{endpoint}")
        return data.to_dict()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)