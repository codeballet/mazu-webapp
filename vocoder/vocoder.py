import os
import re
import glob
import subprocess
import worldvocoder as wv
import soundfile as sf
import librosa
import soundfile as sf
import numpy as np
import time

# Sleep timer
N = 10

# File paths
MP3_DIR = "/soundfiles/"
WAV_DIR = "/soundfiles/"
SYN_DIR = "./syn_files/"
TEST_DIR = "./test_files/"

# Vocoder settings
PITCH = 1.4
DURATION = 1.3

# initialize vocoder
vocoder = wv.World()


####################
# Helper functions #
####################

# Extract integer number from filename function
def extract_number(file):
    base_filename = os.path.basename(file)
    if "_synthesized" in base_filename:
        number = int(base_filename.split("_")[0])
    else:
        number = int(base_filename.split(".")[0])
    return number


# Synthesize file function
def synthesize(file, number):
    print(f"\nStarting processing of file '{file}'")
    # read audio details
    audio, sample_rate = sf.read(file)
    audio = librosa.to_mono(audio)

    # encode audio
    dat = vocoder.encode(sample_rate, audio, f0_method='harvest', is_requiem=True)

    dat = vocoder.scale_pitch(dat, PITCH)
    dat = vocoder.scale_duration(dat, DURATION)

    dat = vocoder.decode(dat)
    output = dat["out"]

    # Assuming 'output' contains your synthesized audio data (numpy array)
    # Set the desired output file path
    output_file_path = f"{SYN_DIR}{number}_synthesized.wav"

    # Write the audio data to the WAV file
    sf.write(output_file_path, output, sample_rate)
    print(f"Processed '{output_file_path}'\n")

def sorter(item):
    # Extract the numeric part from the filename
    match = re.match(r'.*?(\d+).*?', os.path.basename(item))
    if match:
        return int(match.group(1))
    return float('inf')  # Return infinity for non-numeric filenames


#########################################
# Loop to convert and process new files #
#########################################

while True:

    # Collect sorted lists of all soundfiles that already exist
    # Collect and sort mp3 files
    files_mp3 = glob.glob(f"{MP3_DIR}*.mp3")
    files_mp3.sort(key=sorter)  # Sort based on the extracted numeric value

    # Collect and sort wav files
    files_wav = glob.glob(f"{WAV_DIR}*.wav")
    files_wav.sort(key=sorter)  # Sort based on the extracted numeric value

    # Collect and sort synthesized files
    files_syn = glob.glob(f"{SYN_DIR}*_synthesized.wav")
    files_syn.sort(key=sorter)  # Sort based on the extracted numeric value

    # Check for existing wav files
    wav_list = []
    for wav_file in files_wav:
        wav_number = extract_number(wav_file)
        wav_list.append(wav_number)
    wav_list.sort()

    # Convert all mp3 files to wav not done yet
    for mp3_file in files_mp3:
        # Get the number in the mp3 filename
        mp3_number = extract_number(mp3_file)
        # Check if the file is already converted
        # If not, convert and save as wav file
        if mp3_number not in wav_list:
            # TODO: make sure that the entire file is processed. Try with pydub.
            # This could also be cause by starting the transcoding before file is fully created
            # N sleep time is increased to 10 seconds. Try and see if problem persists.
            test_file = f"{TEST_DIR}{mp3_number}.wav"
            subprocess.call(['ffmpeg', '-i', mp3_file, test_file])
            new_file = f"{WAV_DIR}{mp3_number}.wav"
            subprocess.call(['ffmpeg', '-i', mp3_file, new_file])
            print(f"\nConverted '{mp3_file}'\n")

    # Check for existing synthesized files
    syn_list = []
    for syn_file in files_syn:
        syn_number = extract_number(syn_file)
        syn_list.append(syn_number)
    syn_list.sort()

    # Synthesize all wav files not done yet
    for wav_file in files_wav:
        # Get the number in the filename
        wav_number = extract_number(wav_file)
        # Check if the file is already synthesized
        # If not, process file with vocoder
        if wav_number not in syn_list:
            synthesize(wav_file, wav_number)

    time.sleep(N)
