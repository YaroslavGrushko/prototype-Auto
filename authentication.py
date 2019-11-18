# for voice recording: >>>>>>>>>>>>>>>>>>>>>>>>
import sys
from sys import byteorder
from array import array
from struct import pack

import pyaudio
import wave
# for voice recording <<<<<<<<<<<<<<<<<<<<<<<<<

from python_speech_features import mfcc
import scipy.io.wavfile as wav


# for testing KNeighbours
# for array processing
import numpy as np
# # for graphs
# import matplotlib.pyplot as plt
# to work with .csv/txt/etc.
import pandas as pd
# Import LabelEncoder
from sklearn import preprocessing
# # Split the data between the Training Data and Test Data
# from sklearn.model_selection import train_test_split
# Clissifier:>>>>>>>>>>>>>>>>>>>>
# # kneigbours:
# from sklearn.neighbors import KNeighborsClassifier
# xgboost
# import xgboost as xgb
# from xgboost import  XGBClassifier
# Clissifiers <<<<<<<<<<<<<<<<<<<

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# from sklearn.metrics.pairwise import manhattan_distances

# to save model on disk
from joblib import dump, load
# # for clustering
# from sklearn.cluster import KMeans
# # for acuracy score between clusters
# from sklearn.metrics import accuracy_score
# # for chi-beni index
# from sklearn.metrics.pairwise import manhattan_distances
# Local Outlier Factor (LOF) - Neighbours:
from sklearn.neighbors import LocalOutlierFactor

# for random sentence: >>>>>>>
import random
#  for random sentence <<<<<<<
# AUDIO-FILE PARAMETERS >>>>>>>>>>>>>>>>>>>>>>>>>>>
THRESHOLD = 800
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 44100
# by default
FFT_LENGTH = 4096 # default parameter
# FFT_LENGTH = 65536 # parameter from bachelor work
# AUDIO-FILE PARAMETERS <<<<<<<<<<<<<<<<<<<<<<<<<<<

# MFCCS parameters: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# additional parameters from bachloar work
numcep = 24
lowfreq = 20
highfreq = 8000
# MFCCS parameters <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# identification threshold
# IDENT_threshold = 53.0
IDENT_threshold = 1080
# varification threshold
VARIFY_threshold = 83.0

# identification flag
# IDENT_flag = True
# varification flag
# VARIFY_flag = True

# for recording:
from threading import Timer

# RANDOM SENTENCE >>>>>>>>>>>>>>>>>>>>>>>>
nouns = ('бджола', 'ліс', 'мрія', 'мед', "казка", "птахи", "сонце", "лісоруб", "ведмідь", "час", "заєць","вовк")
conj = ("і", "та", "як", "а","і","та", "і", "та", "як", "і")
prep = ("на", "через", "під", "навпростець", "навколо", "над", "у", "в","перед","спереду","перед")
verbs = ("летить", "йде", "танцює", "ллється", "пливе", "грає", "пахне", "світить", "шумить", "сидить") 
adv = ("сухо","дружньо", "тепло", "гарно", "мирно", "природньо", "привітньо", "легко", "розважливо", "поступово", "життєво")
adj = ("веселий", "життєвий", "мальовничий", "зелений", "охайний", "прекрасна", "спокійний","мужній","стриманий", "смішний")
num1 = random.randrange(0,9)
num2 = random.randrange(0,9)
num3 = random.randrange(0,9)
num4 = random.randrange(0,9)
num5 = random.randrange(0,9)
num6 = random.randrange(0,9)
num7 = random.randrange(0,9)
num8 = random.randrange(0,9)
num9 = random.randrange(0,9)
num10 = random.randrange(0,9)
 # let's generate random sentence:
random_sentence = adj[num1]+' '+nouns[num2] + ' ' +prep[num3] +' '+nouns[num4+1]+' '+adv[num5]+' '+conj[num6]+' '+adv[num7+1]+' '+ verbs[num8] +' '+prep[num9+1]+' '+nouns[num10+2]
 # timer>>>>>:
# isSilent flag. whn thiflag is silent, than recording is stop
isRecordingEnded = False

def timeout():
    global isRecordingEnded
    isRecordingEnded=True

    # duration is in seconds
t = Timer(3.0, timeout)
# timer<<<<<<  
# RANDOM SENTENCE <<<<<<<<<<<<<<<<<<<<<<<<<
def is_silent(snd_data):
    "Returns 'True' if below the 'silent' threshold"
    return max(snd_data) < THRESHOLD
    # # let's make it not truncate when it is silent
    # return isSilent

def normalize(snd_data):
    "Average the volume out"
    MAXIMUM = 16384
    times = float(MAXIMUM)/max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r

def trim(snd_data):
    "Trim the blank spots at the start and end"
    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i)>THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # Trim to the left
    snd_data = _trim(snd_data)

    # Trim to the right
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data

def add_silence(snd_data, seconds):
    "Add silence to the start and end of 'snd_data' of length 'seconds' (float)"
    r = array('h', [0 for i in range(int(seconds*RATE))])
    r.extend(snd_data)
    r.extend([0 for i in range(int(seconds*RATE))])
    return r

def record():
    """
    Record a word or words from the microphone and 
    return the data as an array of signed shorts.

    Normalizes the audio, trims silence from the 
    start and end, and pads with 0.5 seconds of 
    blank sound to make sure VLC et al can play 
    it without getting chopped off.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')
    
    while 1:
        # little endian, signed short
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            # this is not useful. just let it be :)
            num_silent += 1
        elif not silent and not snd_started:
            # if now already not silent:
            snd_started = True
            # let's start timer: >>>
            t.start()
            # start timer <<<<

        # let's break recodring
        # if snd_started and num_silent > 30:
        #     break
        if isRecordingEnded:
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    # r = add_silence(r, 0.5)
    return sample_width, r

def record_to_file(path):
    "Records from the microphone and outputs the resulting data to 'path'"
    sample_width, data = record()
    data = pack('<' + ('h'*len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()

# # this function is needed for chi-beni criteria
# #function that gets points from cluster 
# def getPoints(cluster, cluster_map):
#     current_cluster = cluster_map[cluster_map.cluster == cluster]
#     # get appropriate points wich belongs to this cluster
#     points = np.asarray(current_cluster['voiceprints'].tolist(), dtype=np.float16)
#     # cut points (it is needed for manhattan_distances() function)
#     # cutted_points= points[:1000]
#     return points

# ==============================+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Varification
    # check if X_inspect and  X_pred belongs to one cluster
def check_by_Verificator(X_inspect, Y_inspect, X_pred, Y_pred, result_individual, IDENT_flag):
#     # get X_inspect length
#     inspect_len = len(X_inspect)
#     pred_len = len(X_pred)
#     # X_for_clustering = np.append(X_inspect, X_pred)
#     X_for_clustering = np.concatenate((X_inspect, X_pred), axis=0)
#     # let's y_inspect will be "0" and y_pred wil be "1"
#     y_inspect = np.full(inspect_len, 0)
#     y_pred = np.full(pred_len, 1)
#     y_true = np.concatenate((y_inspect, y_pred), axis=0)
#     # let's break into 2 clusters with KMeans
#     kmeans = KMeans(n_clusters=2).fit(X_for_clustering)
#     # # k-means centers
#     # cluster_centers = kmeans.cluster_centers_
#     # k-means classified pixels
#     labels = kmeans.labels_
#     accuracy = accuracy_score(y_true, labels)
#     print("ac-cy of k-means cl-ng (difference): "+ str(accuracy*100)+"%")
  

# # ///////////////////////////////////////////////////////////////////////////////////////////////////
#     # let's find Chi-Beni metrics (inner-cluster distaance/outer claster distance////////////////////
#     # create dataframe with two columns (cluster of point and coordinates of point)
#     cluster_map = pd.DataFrame({'cluster': labels, 'voiceprints': list(X_for_clustering)}, columns=['cluster', 'voiceprints'])

#     # inner distance in cluster
#     inner_cluster_distance = 0
#     # get unique labels (our clusters) from all labels
#     unique_labels = np.unique(labels)

#     # let's calculate inner cluster distance
#     # for any cluster
#     for cluster in unique_labels:
#         points=getPoints(cluster, cluster_map)
#         #calculate distances 
#         md = manhattan_distances(points)
#         # sum all distances
#         sum_md= sum(sum(md))
#         # append result to inner_cluster_distance
#         inner_cluster_distance=inner_cluster_distance+sum_md

#     # let's calculate summery distance between differant clusters////
#     # distance between diffarant clusters
#     distance_between_clusters = 0

#     for cluster1 in unique_labels:
#         for cluster2 in unique_labels:
#             # we need points from different clusters
#             if cluster1!=cluster2:
#                 # cluster1
#                 cutted_pointns1=getPoints(cluster1, cluster_map)
#                 # cluster2
#                 cutted_pointns2=getPoints(cluster2, cluster_map)

#                 #calculate distances 
#                 md = manhattan_distances(cutted_pointns1,cutted_pointns2)
#                 # sum all distances
#                 sum_md= sum(sum(md))
#                 # append result to inner_cluster_distance
#                 distance_between_clusters=distance_between_clusters+sum_md

#     chi_beny=float(inner_cluster_distance/distance_between_clusters)
#     print("chi_beny (similarity): "+str(chi_beny))
#     chi_beny_100 = chi_beny*100


#     if(chi_beny_100<VARIFY_threshold):
#         global VARIFY_flag 
#         VARIFY_flag = False
        
#     if(VARIFY_flag or IDENT_flag):
#         if(VARIFY_flag and IDENT_flag):
#             print("ALLOW")
#         else:
#             print("TRY MORE")
#     else:
#         print("DENY")    
# # //////////////////////////////////////////////////////////////////////////////
# # distance between clusters centers
#     # k-means centers
#     cluster_centers = kmeans.cluster_centers_
#     #calculate distances 
#     md = manhattan_distances(cluster_centers)
#     # sum all distances
#     sum_md = md[0,1]
#     print("b-n-centers distance: "+ str(sum_md))
# ///////////////////////////////////////////////////////////////////////////////////////////////////
    # global VARIFY_flag 
    # global IDENT_flag
    novelty_relation = 0
    #  Local Outlier Factor (LOF)
    # novelty = True / it is to make novelty prediction.
    # when you train clean data and check it with anomaly on new data
    clf = LocalOutlierFactor(n_neighbors=20, algorithm='auto', leaf_size=30, metric='minkowski', p=2, metric_params=None,contamination='legacy', novelty=True, n_jobs=None)
    # clf = IsolationForest(n_estimators=100, max_samples='auto', contamination='legacy', max_features=1.0, bootstrap=False, n_jobs=None, behaviour='old', random_state=None, verbose=0)
    # clf = EllipticEnvelope(random_state=0)
    # let's fit with predict X-value
    clf.fit(X_pred)
    # let's predict with inspect X-value
    Y_clf_pred = clf.predict(X_inspect)
    unique, counts = np.unique(Y_clf_pred, return_counts=True)
    print(dict(zip(unique, counts)))
    # (-1)/1   novelty_relation = not_equels_count/equels_count
    if(len(counts)<=1):
        counts[1]=1
    novelty_relation = counts[1]/counts[0]


    print('NOVELTY_RELATION: '+str('%.3f'%(novelty_relation)))
    if novelty_relation > 0.68:
        VARIFY_flag = 2
    else:
        if novelty_relation < 0.4:
            VARIFY_flag = 0
        else:
            VARIFY_flag = 1

       
# when main method is one of both:
    if(IDENT_flag==2):
        action = "ALLOW"
    else:
        if(IDENT_flag==0):
            action = "DENY"
        # if IDENT_flag == TRY MORE
        else:
            if VARIFY_flag==2:
                action = "ALLOW"
            else:
                if VARIFY_flag==0:
                    action = "DENY"
                # if IDENT_flag == TRY MORE and VARIFY_flag==TRY MORE
                else:
                    action = "TRY MORE"

    # print("action: "+str(action))

# ///////////////////////////////////////////////////////////////////////////////////////////////////
    return result_individual, action 
# ==============================+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Varification
if __name__ == '__main__':
    # recording part
    # print("please, enter your name:")
    # label = input("")
    label = "on_inspection"
    print("прочитайте речення в мікрофон:")
    print(str(random_sentence))
    record_to_file('on_inspection.wav')
    # print("on_inspection: done - result written to on_inspection.wav")

    label = 'on_inspection'
    (rate,sig) = wav.read("on_inspection.wav")
    mfcc_feat = mfcc(sig,rate,winlen=0.094,nfft=FFT_LENGTH, numcep=numcep, lowfreq=lowfreq, highfreq=highfreq)
    # le's print results
    print('\n\n')
    print('============================================================================')
    print('================================results:====================================')
    print('============================================================================')
    print('\n\n')

    print('DURATION: '+str(len(sig)/rate))
    ##text=List of strings to be written to file
    with open('on_inspection_dataset.csv','w+') as file:
        file.write('person, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 \n')
        for line in mfcc_feat:
            strLine = str(label) + ',' + ','.join(map(str, line))
            file.write(strLine)
            file.write('\n')

    # print("on_inspection: mfccs was recorded to .csv")

# LET'S TEST ON_INSPECTION SAMPLE


   # header = None if dataset has no header
    dataset =  pd.read_csv('on_inspection_dataset.csv',sep=',')

    # let's encode catigorial data to numbers
    dataset_encoded = dataset.copy() 
    # creating labelEncoder
    le = preprocessing.LabelEncoder()
    # Converting string labels into numbers.
    dataset_encoded['person']=le.fit_transform(dataset['person'])

    # make 2 datasets X - for input data and Y- for output
    #  input data
   # X_inspect = dataset_encoded.iloc[:, 1:14].values # for default mfccs parameters
    X_inspect = dataset_encoded.iloc[:, 1:25].values #for bachlors work mfccs parametrs
    
    #  output data
    Y_inspect = dataset_encoded.iloc[:, 0].values

# ==============================+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Identification>>>>>>>>>>>>>>>>
    # kneigbours:>>>
    # load the learned model
    clf = load('Classifier_model.joblib') 
    # let's predict 
    # simple sci-learn clissifier
    y_pred = clf.predict(X_inspect)


    counts = np.bincount(y_pred)
    counts_sorted = np.argsort(counts, axis = 0)
    index_of_max_elem = counts_sorted[len(counts) - 1]
    index_of_second_max_elem = counts_sorted[len(counts) - 2]


    max_elem = counts[index_of_max_elem]
    second_max_elem = counts[index_of_second_max_elem]

    
    # let's count percentege between 1st max and 2nd max 
    # must be from [1,9999]
    first_max_second_max_relation = max_elem/second_max_elem
    # ->>>the more so the better!

    # let's reverse from int to marks

    # load labelEncoder
    le = load('LabelEncoder.joblib') 
    # result_mark = le.inverse_transform([max_freq_item])
    result_mark = le.classes_[index_of_max_elem]

    # global current_count
    IDENT_flag =0
    # current_count+=1
    # print(str(current_count))
    # three state: ALLOW(2)/TRY MORE(1)/DENY(0):
    if(first_max_second_max_relation < 1.08):
        IDENT_flag=0 #DENY
    if(1.08 <= first_max_second_max_relation and first_max_second_max_relation < 5):
        IDENT_flag=1 #TRY MORE
    if(first_max_second_max_relation>=5):
        IDENT_flag=2 #ALLOW

    # print('RELATION:'+ str(first_max_second_max_relation))
    # ==============================+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Identification<<<<<<<<<<<<<<<<

    # let's find all X_pred according to result_mark[0]
    demo_dataset = pd.read_csv('demo_dataset.csv') 
    result_individual = result_mark
    pred = demo_dataset.loc[demo_dataset['person'].astype(str)==result_individual]
    # input data of predicted  individual
    # X_pred = pred.iloc[:, 1:14].values #for default mfccs parametrs
    X_pred = pred.iloc[:, 1:25].values #for mfccs parametrs from bachlor's work
    # output data of predicted  individual
    Y_pred = pred.iloc[:, 0].values
    # ==============================+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Varification
    individual, action = check_by_Verificator(X_inspect, Y_inspect, X_pred, Y_pred, result_individual, IDENT_flag)
    print('RELATION: '+ str('%.3f'%(first_max_second_max_relation)))
    print('individual: ' + str(individual)+'\naction: '+str(action))