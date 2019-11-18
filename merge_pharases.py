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


# merge current person-directory into appropriate file
def merge_directory(PERSONS_DIRECTORY_FROM, PERSONS_DIRECTORY_TO, dirname):
    
    all_files = os.listdir(PERSONS_DIRECTORY_FROM+'/'+dirname)
    file_to_merge = AudioSegment.from_file(PERSONS_DIRECTORY_FROM+'/'+dirname+'/'+all_files[0], "flac")

    for filename in  all_files:
        # (rate,sig) = wav.read(directory_from+'/'+filename)
        audio = AudioSegment.from_file(PERSONS_DIRECTORY_FROM+'/'+dirname+'/'+filename, "flac")
        
        file_to_merge = file_to_merge + audio
        
    #Exports to a wav file in the current path.
    file_to_merge.export(PERSONS_DIRECTORY_TO+'/'+dirname+'.wav', format="wav") 



# all persons in main dirname
def merge_all_directories(PERSONS_DIRECTORY_FROM, PERSONS_DIRECTORY_TO):
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

    # get all perdons directories
    all_persons = os.listdir(PERSONS_DIRECTORY_FROM)
    # foreach each person-directory
    for dirname in  all_persons:
        # merge current person-directory into appropriate file
        merge_directory(PERSONS_DIRECTORY_FROM, PERSONS_DIRECTORY_TO, dirname)

merge_all_directories('voices', 'real_voices_texts')