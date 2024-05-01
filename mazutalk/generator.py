import numpy as np
from tensorflow.keras import callbacks


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

