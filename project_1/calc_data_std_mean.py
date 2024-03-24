import json

from tools import calc_data_mean_std

results = calc_data_mean_std("data")
print("Mean:", results["mean"])
print("Std:", results["std"])

with open("data_mean_std.json", "w") as f:
    json.dump(results, f)
