import pandas as pd
import os
import json

PATH = 'results/'
FILE_NAME = 'results.csv'
FILE_EXTENTION = FILE_NAME.split('.')[1]
FILE = os.path.join(PATH, FILE_NAME)

FILE_WO_EXT = FILE_NAME.split('.')[0]
JSON_PATH = os.path.join(PATH, 'avg_' + FILE_WO_EXT + '.json')  

if FILE_EXTENTION == 'csv':
    results = pd.read_csv(FILE)
elif FILE_EXTENTION == 'xlsx':
    results = pd.read_excel(FILE)

metrics = ["mse", "r2", "mae"]
models = ["svr", "pls", "rf"]
variants = ["base", "gan", "shift"]

stats = {}

for model in models:
    stats[model.upper()] = {}
    
    for m in metrics:
        stats[model.upper()][m] = {}
        
        for v in variants:
            col = f"{m}_{model}_{v}"
            
            mean = float(results[col].mean())
            std = float(results[col].std())
            
            stats[model.upper()][m][v] = {
                "mean": mean,
                "std": std
            }

with open(JSON_PATH, "w") as f:
    json.dump(stats, f, indent=4)

