import numpy as np

file_path = "results.npz"

r = np.load(file_path, allow_pickle=False)

new_data = {}
for key in r.files:
        arr = r[key]

        if hasattr(arr, "shape") and arr.shape[0] == 66528:
            new_data[key] = arr[:-1]
        else:
            new_data[key] = arr
np.savez("results_trucated.npz", **new_data)
print("done")

