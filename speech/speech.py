from sqlalchemy import create_engine, text
import time
from gtts import gTTS


time.sleep(15)

# Connect to db
engine = create_engine("postgresql+psycopg://postgres:postgres@db/postgres")


# Initiate db with a last_message_id value as zero
with engine.connect() as conn:
    conn.execute(text(
        "INSERT INTO last (last_message_id) VALUES (0);"
    ))
    conn.commit()
print("Saved value to 'last' table")


################
# Keep looping #
################

# Set sleep timer N
N = 3
while True:
    # Reset tracker and answer variables
    tracker = 0
    answer = None

    # Update tracker with highest value from db table 'last'
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT MAX(last_message_id) FROM last;"
        ))
    for r in result:
        tracker = r[0]

    # Acquire all the answers later than tracker
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT * FROM message WHERE message_id > %i;" % tracker
        ))
    # Extract the values from above query
    for row in result:
        message_id = row[0]
        answer = row[2]
    
        # Update counter
        tracker = row[0]

        # Create new mp3 files if new answers are found
        if answer:
            # First, update the db with tracker value
            with engine.connect() as conn:
                conn.execute(text(
                    "INSERT INTO last (last_message_id) VALUES (:last_message_id);"
                ), [{"last_message_id": tracker}],)
                conn.commit()

            # Truncate answer to be shorter
            # answer_list = answer.split()
            # sliced_list = answer_list[:40]
            # short_answer = ' '.join(sliced_list)
            
            # Generate sound files from texts
            try:
                print(f"Generating Text To sound file from number {message_id}")
                tts = gTTS(answer, lang='sv', slow=True)
            except:
                print("Error: could not generate Text To sound file")

            # Save soundfiles in /soundfiles/ volume
            try:
                print(f"Saving mp3 file of number {message_id}")
                tts.save('/soundfiles/%s.mp3' % message_id)
                # also save for testing in local directory
                tts.save('./mp3_files/%s.mp3' % message_id)
            except:
                print("Error: Could not save mp3 file")
    if not answer:
        print("No new materials for mp3 files")

    time.sleep(N)
