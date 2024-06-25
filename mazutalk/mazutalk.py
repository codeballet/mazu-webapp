import glob
import numpy as np
import json
import re
import string
import datetime
import requests
import os

from sqlalchemy import create_engine, text
import time

import tensorflow as tf
import keras
from tensorflow.keras import layers, models, losses, callbacks

# Import functions from files
from load_data import (
    load_poems,
    load_phd,
    # load_databricks,
    conc_data,
    pad_punctuation
)
from transformer import causal_attention_mask, TransformerBlock
from embedding import TokenAndPositionEmbedding
from generator import TextGenerator


# Server requests
# URL = "http://web:8000/api_mazu/"
URL = "https://spaceengineering.io/api_mazu/"
LOOP = True
N = 20

# Training variables
LOAD_MODEL = True
TRAIN = False
EPOCHS = 2

VOCAB_SIZE = 100000
MAX_LEN = 80
EMBEDDING_DIM = 512
KEY_DIM = 512
N_HEADS = 4
FEED_FORWARD_DIM = 512
VALIDATION_SPLIT = 0.2
SEED = 42
BATCH_SIZE = 32
DATASET_REPETITIONS = 1


# Connect to db
engine = create_engine("postgresql+psycopg://postgres:postgres@db/postgres")


def main():
    # Acquire all the text data
    complete_data = conc_data()

    # Prepare the data
    text_data = [pad_punctuation(x) for x in complete_data]

    # Convert data to Tensorflow dataset
    text_ds = (
        tf.data.Dataset.from_tensor_slices(text_data)
        .batch(BATCH_SIZE)
        .shuffle(1000)
    )

    # Create a vectorisation layer
    vectorize_layer = layers.TextVectorization(
        standardize="lower",
        max_tokens=VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=MAX_LEN + 1,
    )

    # Adapt the layer to the training set
    vectorize_layer.adapt(text_ds)
    vocab = vectorize_layer.get_vocabulary()
       
    # Build Transformer model
    inputs = layers.Input(shape=(None,), dtype=tf.int32)
    x = TokenAndPositionEmbedding(MAX_LEN, VOCAB_SIZE, EMBEDDING_DIM)(inputs)
    x, attention_scores = TransformerBlock(
        N_HEADS, KEY_DIM, EMBEDDING_DIM, FEED_FORWARD_DIM
    )(x)
    outputs = layers.Dense(VOCAB_SIZE, activation="softmax")(x)
    gpt = models.Model(inputs=inputs, outputs=[outputs, attention_scores])
    gpt.compile("adam", loss=[losses.SparseCategoricalCrossentropy(), None])

    # Load weights
    if LOAD_MODEL:
        print("\nLoading weights\n")
        gpt.load_weights("./checkpoint/checkpoint.weights.h5")
        # gpt.load_weights("./checkpoint/checkpoint.ckpt")

    # Create a model save checkpoint
    model_checkpoint_callback = callbacks.ModelCheckpoint(
        filepath="./checkpoint/checkpoint.weights.h5",
        # filepath="./checkpoint/checkpoint.ckpt",
        save_weights_only=True,
        save_freq="epoch",
        verbose=0,
    )

    tensorboard_callback = callbacks.TensorBoard(log_dir="./logs")

    # Instantiate the TextGenerator
    text_generator = TextGenerator(vocab, gpt)

    # Create the training set of words and the same text shifted by one word
    def prepare_inputs(text):
        text = tf.expand_dims(text, -1)
        tokenized_sentences = vectorize_layer(text)
        x = tokenized_sentences[:, :-1]
        y = tokenized_sentences[:, 1:]
        return x, y

    train_ds = text_ds.map(prepare_inputs).repeat(DATASET_REPETITIONS)
    
    # Fit the model
    if TRAIN:
        print("\nTraining...\n")
        gpt.fit(
            train_ds,
            epochs=EPOCHS,
            callbacks=[model_checkpoint_callback, tensorboard_callback, text_generator],
        )
    else:
        print("\nNot training model\n")


    ######################
    # Generate text loop #
    ######################

    if LOOP:

        print("Mazutalk about to start requests...")
        time.sleep(5)

        headers = {
            "Authorization": "Bearer %s" % os.environ.get("BEARER")
        }

        while True:
            print("Inside mazutalk loop")
            try:
                # Get prompts from the web app
                response = requests.get(URL, headers=headers)
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

            try:
                # If new data, step through the prompts and generate answers
                if len(data["messages"]) > 0:
                    for message in data['messages']:
                        print(f"message:\n{message}")
                        message_id = int(message['pk'])
                        session_key = message['fields']['prompt']
                        prompt = message['fields']['prompt']
                        
                        answer_list = text_generator.talk(
                            prompt, max_tokens=MAX_LEN, temperature=0.5
                        )
                        answer_raw = ' '.join(map(str, answer_list))
                        print(answer_raw)

                        # Save to db
                        with engine.connect() as conn:
                            print("mazutalk ai saving to database")

                            conn.execute(
                                text("INSERT INTO message (message_id, prompt, answer) VALUES (:message_id, :prompt, :answer);"), 
                                [{"message_id": message_id, "prompt": prompt, "answer": answer_raw}],
                            )
                            conn.commit()

                        try:
                            # Send POST request back to the web app
                            payload = {
                                "id": message_id,
                                "session_key": session_key,
                                "prompt": prompt,
                                "answer": answer_raw,
                            }
                            response = requests.post(URL, headers=headers, data=payload)
                            print(f"POST response: {response}")
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

            except:
                print("Error: mazutalk cannot connect to db")

            time.sleep(N)

    
if __name__ == "__main__":
    main()