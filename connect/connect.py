from sqlalchemy import create_engine, text
import time
import requests
import os
import json

time.sleep(5)

# Connect to db
engine = create_engine("postgresql+psycopg://postgres:postgres@db/postgres")


print("Starting connect...")


# Define request variables
url = "http://web:8000/api_mazu/"
# url = "https://spaceengineering.io/api_mazu/"
headers = {
    "Authorization": "Bearer %s" % os.environ.get("BEARER")
}


while True:
    # Get prompts from the web app
    try:
        payload = {}
        response = requests.get(url, headers=headers, json=payload)
        print(f"GET response code: {response.status_code}")

        data = response.json()
        print(f"Response:\n{data}")
        print(len(data["messages"]))
    except requests.exceptions.HTTPError as errh:
        print ("Http Error:",errh)
    except requests.exceptions.ConnectionError as errc:
        print ("Error Connecting:",errc)
    except requests.exceptions.Timeout as errt:
        print ("Timeout Error:",errt)
    except requests.exceptions.RequestException as err:
        print ("OOps: Something Else",err)

    # If new data, step through the prompts and generate sentences
    if len(data["messages"]) > 0:
        counter = 0
        for message in data['messages']:
            print(f"message:\n{message}")
            creation = int(message['pk'])
            prompt = message['fields']['prompt_text']
            print(f"Prompt: {prompt}")
            
            # Create a fake Mazu response for testing
            sentence_raw = f"Fake Mazu answer {counter}"
            counter += 1
            print(sentence_raw)

            # Save to db
            # with engine.connect() as conn:
            #     print("mazutalk ai saving to database")

            #     conn.execute(
            #         text("INSERT INTO speech (creation, prompt, sentence) VALUES (:creation, :prompt, :sentence);"), 
            #         [{"creation": creation, "prompt": prompt_text, "sentence": sentence_raw}],
            #     )
            #     conn.commit()

            # Send POST request back to the web app
            payload = {
                "id": message['pk'],
                "session_key": message['fields']['session_key'],
                "prompt": prompt,
                "answer": sentence_raw,
            }

            # requests.post(url, headers=headers, data=payload)
            try:
                response = requests.post(url, headers=headers, data=payload)
                print(f"POST response code: {response.status_code}")
            except requests.exceptions.HTTPError as errh:
                print ("Http Error:",errh)
            except requests.exceptions.ConnectionError as errc:
                print ("Error Connecting:",errc)
            except requests.exceptions.Timeout as errt:
                print ("Timeout Error:",errt)
            except requests.exceptions.RequestException as err:
                print ("OOps: Something Else",err)

    else:
        print("mazutalk received no new data")

    time.sleep(10)
