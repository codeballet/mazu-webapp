from sqlalchemy import create_engine, text
import time
from gtts import gTTS
# import datetime

# Timer interval value
N = 5

print("Speech app started")
time.sleep(N)

# Connect to db
engine = create_engine("postgresql+psycopg://postgres:postgres@db/postgres")


# Acquire db data every Nth second
while True:
    print("Speech app loop")
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT * FROM speech;"
        ))

    for row in result:
        created = row[0]
        sentence = row[2]
        print(f"created: {created}")
        print(f"Sentence: {sentence}")

        # Get a unique timestamp for filenaming
        # timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')

        # Generate speech from sentences and save as mp3 file
        try:
            print("Generating Text To Speech...")
            tts = gTTS(sentence, lang='sv', slow=True)
        except:
            print("Error: could not generate Text To Speech")
        try:
            print("Saving mp3 file")
            tts.save('./tts/%s.mp3' % created)
        except:
            print("Error: Could not save mp3 file")

    time.sleep(N)
