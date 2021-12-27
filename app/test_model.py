import json
import requests
import numpy as np

if __name__ == "__main__":
    X = np.random.uniform(0, 1, (1, 224, 224, 1))
    input_data = json.dumps({
        "signature_name": "serving_default",
        "instances": X.tolist(),
    })

    SERVER_URL = 'http://localhost:8501/v1/models/forest_classification:predict'

    response = requests.post(SERVER_URL, data=input_data)
    response.raise_for_status() # raise an exception in case of error
    response = response.json()

    print(response)
