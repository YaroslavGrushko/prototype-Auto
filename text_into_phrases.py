# for reading .wav-file
import scipy.io.wavfile as wav
from pydub import AudioSegment
# for work with directories
import os
# for remove directory
import shutil

def delete_folder_content(folder):
    # folder = '/path/to/folder'
    try:
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)
    except Exception as e:
                print(e)


def split_file(directory_from, PERSONS_DIRECTORY_TO, filename, time):
    (rate,sig) = wav.read(directory_from+'/'+filename)
    audio = AudioSegment.from_wav(directory_from+'/'+filename)
    #splice frames to get a list strings each representing a 'time' length
    #wav file

    # duration of whole .wav-file
    duration = len(sig)/rate
    durationMilisec = duration*1000
    # time in miliseconds
    timeMilisec = time*1000
    x=0
    
    # let's find index of char that is near .wav
    index = filename.find('.wav')
    directory = filename.replace('.wav','')
    directory = PERSONS_DIRECTORY_TO+'/'+directory

    if not os.path.exists(directory):
        os.makedirs(directory)

    while x+timeMilisec<=durationMilisec:
        # nea audio frame
        newAudio= audio[x:x+timeMilisec]
        # create newAudio filename
        newAudio_filename = directory+'/'+filename[:index] + str(int(x)) + filename[index:]
        #Exports to a wav file in the current path.
        newAudio.export(newAudio_filename, format="wav") 
        # iterate x
        x=x+timeMilisec


# all persons in main dirname
def split_all_files(PERSONS_DIRECTORY_FROM, PERSONS_DIRECTORY_TO, TIME_TO_SPLIT):
    # # PERSONS_DIRECTORY_TO = 'real_voices'
    # try:
    #     # delete old directory
    #     shutil.rmtree(PERSONS_DIRECTORY_TO, ignore_errors=True)
    # except:
    #     print("An exception occurred")
        
    delete_folder_content(PERSONS_DIRECTORY_TO)
    # create new directory
    if not os.path.exists(PERSONS_DIRECTORY_TO):
        os.makedirs(PERSONS_DIRECTORY_TO)

    # PERSONS_DIRECTORY_FROM = 'real_voices_texts'


    # TIME_TO_SPLIT = 1.0

    all_persons = os.listdir(PERSONS_DIRECTORY_FROM)
    for filename in  all_persons:
        split_file(PERSONS_DIRECTORY_FROM, PERSONS_DIRECTORY_TO, filename, TIME_TO_SPLIT)