import kagglehub
import pandas as pd
import os
import random
import glob

path = kagglehub.dataset_download("jacksoncrow/stock-market-dataset")

dir_path = 'datasets_chosen'

csv_files = glob.glob(dir_path + "/*.csv")
for csv_file in csv_files:
    os.remove(csv_file)
    print(f"Deleted {csv_file}")

files = os.listdir(path + "/stocks")

randomized_files = random.sample(files, 100)
print(randomized_files)

for files in randomized_files:
    df = pd.read_csv(path + "/stocks/" + files)
    outPath = os.path.join(dir_path, files)
    df = df.dropna()

    df.to_csv(outPath, index=False, columns=["Date","Close", "Volume"])