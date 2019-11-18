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
# Classifiers>>>>>>>:

# kneigbours:
from sklearn.neighbors import KNeighborsClassifier
# mlp
from sklearn.neural_network import MLPClassifier
# svm
from sklearn.svm import SVC
# # GPC:
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF

# Decision Tree Classifier:
from sklearn.tree import DecisionTreeClassifier
# Naive Bayes:
from sklearn.naive_bayes import GaussianNB
# AdaBoost
from sklearn.ensemble import AdaBoostClassifier
# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
# Classifiers <<<<<<
# to save model on disk
from joblib import dump, load

# Second Varificator >>>>>>>>>>:
# Local Outlier Factor (LOF) - Neighbours:
from sklearn.neighbors import LocalOutlierFactor
# One Class SVM:
from sklearn.svm import OneClassSVM
# Isolation Forest:
from sklearn.ensemble import IsolationForest
# Second Varificator <<<<<<<<<<

# ABOUT DATASET>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# initial dataset dir
PERSONS_DIRECTORY_FROM = 'real_voices_texts'
# dataset directory
dataset_directory = 'real_voices'
# dictors:
# dataset_directory = 'voices'

# number of persons to learn (at all = 40):
number_of_persons_to_learn = 30
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

# Identification thesholds:
# for k-nn:
maxIdent = 5
# maxIdent = 99999
minIdent = 2
# minIdent = 1.2

# # vithout 2nd verificator
# # for k-nn:
# maxIdent = 2.1
# minIdent = 2.1


# Verification thesholds:
# for k-nn:
maxVerif = 0.4
minVerif = 0.2


