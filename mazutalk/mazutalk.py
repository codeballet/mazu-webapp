import glob
import numpy as np
import json
import re
import string
import datetime
import requests

from sqlalchemy import create_engine, text
import time

import tensorflow as tf
import keras
from tensorflow.keras import layers, models, losses, callbacks


#############
# Constants #
#############

VOCAB_SIZE = 100000
MAX_LEN = 80
EMBEDDING_DIM = 512
KEY_DIM = 512
N_HEADS = 4
FEED_FORWARD_DIM = 512
VALIDATION_SPLIT = 0.2
SEED = 42
LOAD_MODEL = True
BATCH_SIZE = 32
EPOCHS = 1
DATASET_REPETITIONS = 1
TRAIN = False


#################
# Connect to db #
#################
engine = create_engine("postgresql+psycopg://postgres:postgres@db/postgres")


#############
# Load data #
#############

# Load Swedish translations of Chinese poems
def load_poems():
    file_list = glob.glob("./data/zh_poems_sv/*.json")

    # Put the file contents in a list
    translations_sv = []
    for file in file_list:
        with open(file, 'r') as f:
            for poem in json.load(f):
                translations_sv.append(poem)

    return translations_sv

# Load Swedish translation of my PhD
def load_phd():
    with open("./data/stjernholm-texts/phd_thesis_sv.json", 'r') as f:
        translation_phd = json.load(f)

    return translation_phd

# Load Swedish translation of Databricks dataset
def load_databricks():
    with open("/app/data/databricks/databricks-dolly-15k-sv.json", 'r') as f:
        translation_bricks = json.load(f)

    return translation_bricks

# Concatenate all the data to one list
def conc_data():
    return load_poems() + load_phd() + load_databricks()


################
# Prepare data #
################

# Pad the punctuation, to treat them as separate 'words'
def pad_punctuation(s):
    s = re.sub(f"([{string.punctuation}, '\n'])", r" \1 ", s)
    s = re.sub(" +", " ", s)
    s = re.sub('"', '', s)
    s = re.sub(' * ', ' ', s)
    return s


#########################
# Transformer functions #
#########################

# Causal Attention Mask function
def causal_attention_mask(batch_size, n_dest, n_src, dtype):
    i = tf.range(n_dest)[:, None]
    j = tf.range(n_src)
    m = i >= j - n_src + n_dest
    mask = tf.cast(m, dtype)
    mask = tf.reshape(mask, [1, n_dest, n_src])
    mult = tf.concat(
        [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
    )
    return tf.tile(mask, mult)

# Transformer Block custom layer
class TransformerBlock(layers.Layer):
    def __init__(self, num_heads, key_dim, embed_dim, ff_dim, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.attn = layers.MultiHeadAttention(
            num_heads, key_dim, output_shape=embed_dim
        )
        self.dropout_1 = layers.Dropout(self.dropout_rate)
        self.ln_1 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn_1 = layers.Dense(self.ff_dim, activation="relu")
        self.ffn_2 = layers.Dense(self.embed_dim)
        self.dropout_2 = layers.Dropout(self.dropout_rate)
        self.ln_2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = causal_attention_mask(
            batch_size, seq_len, seq_len, tf.bool
        )
        attention_output, attention_scores = self.attn(
            inputs,
            inputs,
            attention_mask=causal_mask,
            return_attention_scores=True,
        )
        attention_output = self.dropout_1(attention_output)
        out1 = self.ln_1(inputs + attention_output)
        ffn_1 = self.ffn_1(out1)
        ffn_2 = self.ffn_2(ffn_1)
        ffn_output = self.dropout_2(ffn_2)
        return (self.ln_2(out1 + ffn_output), attention_scores)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "key_dim": self.key_dim,
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config

# Token and Position Embedding
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, max_len, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.pos_emb = layers.Embedding(input_dim=max_len, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "max_len": self.max_len,
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
            }
        )
        return config


######################
# Train and generate #
######################

# Create a TextGenerator checkpoint
class TextGenerator(callbacks.Callback):
    # def __init__(self, index_to_word, top_k=10):
    def __init__(self, index_to_word, model, top_k=10):
        self.index_to_word = index_to_word
        self.word_to_index = {
            word: index for index, word in enumerate(index_to_word)
        }
        self.model = model # New line

    def sample_from(self, probs, temperature):
        probs = probs ** (1 / temperature)
        probs = probs / np.sum(probs)
        return np.random.choice(len(probs), p=probs), probs

    def generate(self, start_prompt, max_tokens, temperature):
        start_tokens = [
            self.word_to_index.get(x, 1) for x in start_prompt.split()
        ]
        sample_token = None
        info = []
        while len(start_tokens) < max_tokens and sample_token != 0:
            x = np.array([start_tokens])
            y, att = self.model.predict(x, verbose=0)
            sample_token, probs = self.sample_from(y[0][-1], temperature)
            info.append(
                {
                    "prompt": start_prompt,
                    "word_probs": probs,
                    "atts": att[0, :, -1, :],
                }
            )
            start_tokens.append(sample_token)
            try:
                start_prompt = start_prompt + " " + self.index_to_word[sample_token]
            except:
                print("IndexError: list index out of range")
        print(f"\ngenerated text:\n{start_prompt}\n")
        return info

    def talk(self, start_prompt, max_tokens, temperature):
        start_tokens = [
            self.word_to_index.get(x, 1) for x in start_prompt.split()
        ]
        sample_token = None
        sentence = []
        while len(start_tokens) < max_tokens and sample_token != 0:
            x = np.array([start_tokens])
            y, att = self.model.predict(x, verbose=0)
            sample_token, probs = self.sample_from(y[0][-1], temperature)
            start_tokens.append(sample_token)
            new_word = self.index_to_word[sample_token]
            sentence.append(new_word)
        return sentence

    def on_epoch_end(self, epoch, logs=None):
        self.generate("Vatten", max_tokens=80, temperature=1.0)


#################
# Main function #
#################

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

    print("Mazutalk about to start requests...")
    time.sleep(10)

    # Keep looking for new input from the webserver every Nth second
    # Sleep timer
    N = 5
    while True:
        print("Inside mazutalk loop")
        try:
            # Get prompts from the web app
            response = requests.get('http://web:8000/api_mazu')
            data = response.json()
            print(f"Response:\n{data}")
            print(len(data["prompts"]))

            if len(data["prompts"]) > 0:
            # Step through the prompts and generate AI responses
                for prompt in data['prompts']:
                    print(f"prompt:\n{prompt}")
                    creation = int(prompt['pk'])
                    prompt_text = prompt['fields']['prompt_text']
                    
                    sentence_list = text_generator.talk(
                        prompt_text, max_tokens=MAX_LEN, temperature=0.5
                    )
                    sentence_raw = ' '.join(map(str, sentence_list))
                    print(sentence_raw)

                    # Save to db
                    with engine.connect() as conn:
                        print("mazutalk ai saving to database")

                        # conn.execute(
                        #     text("INSERT INTO speech (created, prompt, sentence) VALUES ('54542', 'something', 'ai text generated');")
                        # )

                        conn.execute(
                            text("INSERT INTO speech (creation, prompt, sentence) VALUES (:creation, :prompt, :sentence);"), 
                            [{"creation": creation, "prompt": prompt_text, "sentence": sentence_raw}],
                        )
                        conn.commit()
            else:
                print("mazutalk received no new data")

        except:
            print("Error: mazutalk cannot connect to web app or db")

        time.sleep(N)

    
if __name__ == "__main__":
    main()