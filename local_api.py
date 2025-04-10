import json

import requests
from loguru import logger


# TODO: send a GET using the URL http://127.0.0.1:8000
BASE_URL = "http://127.0.0.1:8000"
r = requests.get(BASE_URL)  # Your code here

# TODO: print the status code
# print()
logger.info(f"Status code: {r.status_code}")
# TODO: print the welcome message
# print()
logger.info(f"Welcome message: {r.json()}")


data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# logger.info(data)

# TODO: send a POST using the data above
req = requests.post(f"{BASE_URL}/data/", json=data)  # Your code here

# TODO: print the status code
# print()
logger.info(f"Status code: {req.status_code}")
# TODO: print the result
# print()
logger.info(f"Result: {req.json()}")
