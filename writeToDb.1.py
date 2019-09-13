# for voice recording: >>>>>>>>>>>>>>>>>>>>>>>>
import sys
from sys import byteorder
from array import array
from struct import pack

import pyaudio
import wave
# for voice recording <<<<<<<<<<<<<<<<<<<<<<<<<

from python_speech_features import mfcc
# from python_speech_features import logfbank
import scipy.io.wavfile as wav

# for work with directories
import os
# for .flac to .wav converting
from pydub import AudioSegment

# to work with .csv/txt/etc.
import pandas as pd
# Import LabelEncoder
from sklearn import preprocessing
# Classifier:
# kneigbours:
from sklearn.neighbors import KNeighborsClassifier
# ABOUT DATASET>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# dataset directory
dataset_directory = 'real_voices'
# number of persons to learn (at all = 2):
number_of_persons_to_learn = 2

# minimal number of records per each person = 55

# sequence number of record from start FROM
# //////////////////////
number_to_learn_FROM = 0
# //////////////////////

# number of records to learn:
number_to_learn = 3
# sequence number of record to learn TO
number_to_learn_TO = number_to_learn_FROM+number_to_learn

# avout dataset <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# ABOUT AUDIO FILE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
THRESHOLD = 500
CHUNK_SIZE = 1024
# mfccs parameters:
# by default
FFT_LENGTH = 4096
# FFT_LENGTH = 65536  #IN BACHLORE WORK
FORMAT = pyaudio.paInt16
RATE = 44100
# ABOUT AUDIO FILE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# additional parameters from bachloar work >>>>>>
numcep = 24
lowfreq = 20
highfreq = 8000
# from bachlor work <<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# VOICE-FILE RECORDING:<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

if __name__ == '__main__':
    
    # create dataset.csv with mfccs header
    with open('demo_dataset.csv','w') as file:
        # strLine = 'person, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13'
        strLine = 'person, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24'
        file.write(strLine)
        file.write('\n')

    # all persons in main dirname
    all_persons = os.listdir(dataset_directory)
    # persons that we select to write to Db
    persons_to_learn = all_persons[:number_of_persons_to_learn]

    # let's write each person to Db
    for person_dir in persons_to_learn:
        full_person_dir = os.path.join(dataset_directory, person_dir)
        # all audio-files in current person's folder
        files = [os.path.join(full_person_dir,f) for f in os.listdir(full_person_dir) if os.path.isfile(os.path.join(full_person_dir, f))]
        # selected audio files from person's folder
        files_new = files[number_to_learn_FROM:number_to_learn_TO]

        # let's foreach every file from person's folder
        for demo in files_new:

            # PUT THEIR NAME OF CURRENT INDIVIDUAL SAMPLE
            LABEL = person_dir
            # read .wav file
            (rate,sig) = wav.read(demo)
            # extract mfccs from demo.wav
            mfcc_feat = mfcc(sig,rate,winlen=0.094,nfft=FFT_LENGTH, numcep=numcep, lowfreq=lowfreq, highfreq=highfreq) #bachlor parameter
            # mfcc_feat = mfcc(sig,rate,winlen=0.094,nfft=FFT_LENGTH, numcep=numcep) #bachlor parameters
            # mfcc_feat = mfcc(sig,rate,winlen=0.094,nfft=FFT_LENGTH) #default parameters          

            ##text=List of strings to be written to file
            with open('demo_dataset.csv','a') as file:
                for line in mfcc_feat:
                    strLine = str(LABEL) + ',' + ','.join(map(str, line))
                    file.write(strLine)
                    file.write('\n')
        # message to terminal after every person is recorded to DB-file
        print("mfccs of "+str(person_dir)+" was recorded to .csv")
    
    # learning:
    # read DB-file with mfccs:
    # header = None if dataset has no header
    dataset =  pd.read_csv('demo_dataset.csv',sep=',')

    # let's encode catigorial data to numbers
    dataset_encoded = dataset.copy() 
    # creating labelEncoder
    le = preprocessing.LabelEncoder()
    # Converting string labels into numbers.
    dataset_encoded['person']=le.fit_transform(dataset['person'].astype(str))
    # # let's save LabelEncoder on disk
    # dump(le, 'LabelEncoder.joblib')

    # make 2 datasets X - for input data and Y- for output
    #  input data
    # X = dataset_encoded.iloc[:, 1:14].values  #for default mfccs parameters
    X = dataset_encoded.iloc[:, 1:25].values # for bachloar mfccs parametrs

    #  output data
    Y = dataset_encoded.iloc[:, 0].values

    # CLASSIFIER: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # KNeighbours >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    clf = KNeighborsClassifier(n_neighbors=3)

    # << KNeighbours <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # let's train:
    clf.fit(X, Y) 
    # model was trained
    print("model was trained")