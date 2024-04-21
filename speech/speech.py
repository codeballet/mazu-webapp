from sqlalchemy import create_engine, text
import time
from gtts import gTTS


time.sleep(15)

# Connect to db
engine = create_engine("postgresql+psycopg://postgres:postgres@db/postgres")


# Initiate db with a last_speech value as zero
try:
    with engine.connect() as conn:
        conn.execute(text(
            "INSERT INTO last (last_speech) VALUES (0);"
        ))
        conn.commit()
    print("Saved value to 'last' table")
except:
    print("Could not initiate 'last' table")


################
# Keep looping #
################

# Set sleep timer N
N = 10
while True:
    # Reset last_creation counter and sentence variables
    last_creation = 0
    sentence = None

    # Update 'last_speech' with highest value from db table 'last'
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT MAX(last_speech) FROM last;"
        ))
    for r in result:
        last_creation = r[0]

    # Acquire all the sentences later than counter last_speech
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT * FROM speech WHERE creation > %i;" % last_creation
        ))
    # Extract the values from above query
    for row in result:
        creation = row[0]
        sentence = row[2]
    
        # Update counter
        last_creation = row[0]

        # Create new mp3 files if new sentences are found
        if sentence:
            # First, update the db with last_speech value
            with engine.connect() as conn:
                conn.execute(text(
                    "INSERT INTO last (last_speech) VALUES (:last_speech);"
                ), [{"last_speech": last_creation}],)
                conn.commit()

            # Truncate sentence to be shorter
            sentence_list = sentence.split()
            sliced_list = sentence_list[:40]
            short_sentence = ' '.join(sliced_list)
            
            # Generate speech from text sentences
            try:
                print(f"Generating Text To Speech from number {creation}")
                tts = gTTS(short_sentence, lang='sv', slow=True)
            except:
                print("Error: could not generate Text To Speech")

            # Save speech as soundfiles in /soundfiles/ volume
            try:
                print(f"Saving mp3 file of number {creation}")
                tts.save('/soundfiles/%s.mp3' % creation)
                # also save for testing in local directory
                tts.save('./mp3_files/%s.mp3' % creation)
            except:
                print("Error: Could not save mp3 file")
    if not sentence:
        print("No new materials for speech mp3 files creation")

    time.sleep(N)
