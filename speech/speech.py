import datetime

# Get a unique timestamp for filenaming
timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')

# with open("./sentences/%s.txt" % timestamp, "w") as text_file:
#     text_file.write("%s" % sentence_raw)

# # # Generate speech and save as mp3 file
# try:
#     print("Text To Speech...")
#     tts = gTTS(sentence_raw, lang='sv', slow=True)
# except:
#     print("Error: could not generate Text To Speech")
# try:
#     print("Saving mp3 file")
#     tts.save('./tts/%s_sv.mp3' % timestamp)
#     # tts.save('./tts/test.mp3')
# except:
#     print("Error: Could not save mp3 file")