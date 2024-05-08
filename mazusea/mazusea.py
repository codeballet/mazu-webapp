import time
import requests
import os
import json


# Wait for web app to start
time.sleep(5)


# Define request variables
url = "http://web:8000/api_sea/"
# url = "https://spaceengineering.io/api_sea/"
headers = {
    "Authorization": "Bearer %s" % os.environ.get("BEARER")
}


# Check for voting results
try:
    response = requests.get(url, headers=headers)
    print(f"GET response code: {response.status_code}")

    data = response.json()
    print(f"Response:\n{data}")
except requests.exceptions.HTTPError as errh:
    print ("Http Error:",errh)
except requests.exceptions.ConnectionError as errc:
    print ("Error Connecting:",errc)
except requests.exceptions.Timeout as errt:
    print ("Timeout Error:",errt)
except requests.exceptions.RequestException as err:
    print ("OOps: Something Else",err)