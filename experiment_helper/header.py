# for voice recording: >>>>>>>>>>>>>>>>>>>>>>>>
import sys
from sys import byteorder
from array import array
from struct import pack

# for array processing
import numpy as np
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
# to save model on disk
from joblib import dump, load
# Varificator:
# Local Outlier Factor (LOF) - Neighbours:
from sklearn.neighbors import LocalOutlierFactor

# ABOUT DATASET>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# initial dataset dir
PERSONS_DIRECTORY_FROM = 'real_voices_texts'
# dataset directory
dataset_directory = 'real_voices'
# dictors:
# dataset_directory = 'voices'

# number of persons to learn (at all = 3):
number_of_persons_to_learn = 40
# import text_into_phrases
from text_into_phrases import split_all_files
# ABOUT DATASET <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ABOUT AUDIO FILE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
THRESHOLD = 500
CHUNK_SIZE = 1024
# mfccs parameters:
# # by default
# FFT_LENGTH = 4096
# FFT_LENGTH = 65536  #IN BACHLORE WORK
FORMAT = pyaudio.paInt16
RATE = 44100
# ABOUT AUDIO FILE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# additional parameters from bachloar work >>>>>>
numcep = 24
lowfreq = 20
highfreq = 8000
# from bachlor work <<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# experimant settings: >>>>>>>>>>>>>>>>>>>>>>>>>>
# MAX_inner_expariment_count = 300
MAX_inner_experiment_count = 70
src_results_folder = "results_folder/"
# experiment settings <<<<<<<<<<<<<<<<<<<<<<<<<<<
