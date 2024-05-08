from sqlalchemy import create_engine, text
import time
import requests
import os
import json


#######################################
# Test program to connect to mazu api #
#######################################

# Wait for web app to start
time.sleep(5)

# Connect to db
engine = create_engine("postgresql+psycopg://postgres:postgres@db/postgres")

# Define request variables
url = "http://web:8000/api_mazu/"
# url = "https://spaceengineering.io/api_mazu/"
headers = {
    "Authorization": "Bearer %s" % os.environ.get("BEARER")
}


# Keep checking for new prompts from web app
while True:
    try:
        response = requests.get(url, headers=headers)
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

    # If new data, step through the prompts and generate answers
    if len(data["messages"]) > 0:
        counter = 0
        for message in data['messages']:
            print(f"message:\n{message}")
            message_id = int(message['pk'])
            session_key = message['fields']['session_key']
            prompt = message['fields']['prompt']
            print(f"Prompt: {prompt}")
            
            # Create a fake Mazu response for testing
            answer_raw = f"Fake Mazu answer {counter}"
            counter += 1
            print(answer_raw)

            # Save to db
            # with engine.connect() as conn:
            #     print("mazutalk ai saving to database")

            #     conn.execute(
            #         text("INSERT INTO message (message_id, prompt, answer) VALUES (:message_id, :prompt, :answer);"), 
            #         [{"message_id": message_id, "prompt": prompt, "answer": answer_raw}],
            #     )
            #     conn.commit()

            # Send POST request back to the web app
            payload = {
                "id": message_id,
                "session_key": session_key,
                "prompt": prompt,
                "answer": answer_raw,
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
