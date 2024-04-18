from sqlalchemy import create_engine, text
import time
from gtts import gTTS


print("Speech app started")
time.sleep(15)

# Connect to db
engine = create_engine("postgresql+psycopg://postgres:postgres@db/postgres")

try:
    with engine.connect() as conn:
        conn.execute(text(
            "INSERT INTO last (last_speech) VALUES (0);"
        ))
        conn.commit()
    print("Saved value to 'last' table")
except:
    print("Could not initiate 'last' table")

# Keep looping to acquire db data
# Sleep timer
N = 10
while True:
    print("Speech app loop")

    # Set counter and sentence variables
    last_creation = 0
    sentence = None

    # Acquire 'last_speech' value from db table 'last'
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT MAX(last_speech) FROM last;"
        ))
    print(result)
    # print(f"max: {max(result)}")
    for r in result:
        # Update counter
        last_creation = r[0]
        number = r[0]

    # Acquire all the speeches
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT * FROM speech WHERE creation > %i;" % last_creation
        ))

    print(f"Speech result: {result}")
    for row in result:
        creation = row[0]
        sentence = row[2]
    
        # Update counter
        last_creation = row[0]

        # Create new mp3 files if new material is found
        if sentence:
            with engine.connect() as conn:
                conn.execute(text(
                    "INSERT INTO last (last_speech) VALUES (:last_speech);"
                ), [{"last_speech": last_creation}],)
                conn.commit()

                # Generate speech from sentences and save as mp3 file
                try:
                    print(f"Generating Text To Speech from number {creation}")
                    tts = gTTS(sentence, lang='sv', slow=True)
                except:
                    print("Error: could not generate Text To Speech")
                try:
                    print(f"Saving mp3 file of number {creation}")
                    tts.save('./tts/%s.mp3' % creation)
                except:
                    print("Error: Could not save mp3 file")
    if not sentence:
        print("No new materials for mp3 files creation")

    time.sleep(N)
