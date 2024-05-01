import glob
import json
import re
import string


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


# Pad the punctuation, to treat them as separate 'words'
def pad_punctuation(s):
    s = re.sub(f"([{string.punctuation}, '\n'])", r" \1 ", s)
    s = re.sub(" +", " ", s)
    s = re.sub('"', '', s)
    s = re.sub(' * ', ' ', s)
    return s
