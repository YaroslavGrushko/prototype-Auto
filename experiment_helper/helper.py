from experiment_helper.header import *
# VARIFY_flag = True
# IDENT_flag = True


# Varification#2 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def check_by_Verificator(X_inspect, Y_inspect, X_pred, Y_pred, result_individual, IDENT_flag):
# VERIFICATOR >>>>>>>>
# # ///////////////////////////////////////////////////////////////////////////////////////////////////
#    set to zero
    novelty_relation = 0
# //////////////////////////////////////////////////////////////////////////////

#   Second Verificator:>>>>>>>>>>  
#  
    #  Local Outlier Factor (LOF)
    # novelty = True / it is to make novelty prediction.
    # when you train clean data and check it with anomaly on new data
    clf = LocalOutlierFactor(n_neighbors=20, algorithm='auto', leaf_size=30, metric='minkowski', p=2, metric_params=None,contamination='legacy', novelty=True, n_jobs=None)
    
    # #   One Class SVM
    # clf = OneClassSVM(gamma='auto')

    # #    Isolation Forest
    # clf = IsolationForest(n_estimators=100, max_samples='auto', contamination='legacy', max_features=1.0, bootstrap=False, n_jobs=None, behaviour='old', random_state=None, verbose=0)
   
    #  Second Verificator <<<<<<<  

   
   
    # let's fit with predict X-value
    clf.fit(X_pred)
    # let's predict with inspect X-value
    Y_clf_pred = clf.predict(X_inspect)
    unique, counts = np.unique(Y_clf_pred, return_counts=True)
    print(dict(zip(unique, counts)))
    # (-1)/1   novelty_relation = not_equels_count/equels_count
    if(len(counts)>1):
        novelty_relation = counts[1]/(counts[0]+counts[1])
    else:
        novelty_relation = 9999



    print('novelty_relation: '+str(novelty_relation))
    if novelty_relation >= maxVerif:
        VARIFY_flag = 2
    else:
        if novelty_relation < minVerif:
            VARIFY_flag = 0
        else:
            VARIFY_flag = 1
# <<<<<<< VERIFICATOR
  
# when main method is one of both:
    if(IDENT_flag==2):
        action = "ALLOW"
    else:
        if(IDENT_flag==0):
            action = "DENY"
        # if IDENT_flag == TRY MORE
        else:#comment below to turn off second verificator
            if VARIFY_flag==2:
                action = "ALLOW"
            else:
                if VARIFY_flag==0:
                    action = "DENY"
                # if IDENT_flag == TRY MORE and VARIFY_flag==TRY MORE
                else:
                 action = "TRY MORE"

    print("action: "+str(action))

    return result_individual, action 
# Varification#2 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# SINGLE-PERSON AUTHENTICATION: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def authentification(full_person_dir,number_from_end_TO,number_from_end_FROM, FFT_LENGTH):

    # get all demo files from full_person_dir:
    files = [os.path.join(full_person_dir,f) for f in os.listdir(full_person_dir) if os.path.isfile(os.path.join(full_person_dir, f))]
    files_count = len(files)

    # BOUNDERIES FOR FILES: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    current_sequence_number_FROM =  files_count-number_from_end_TO-1
    current_sequence_number_TO =  files_count-number_from_end_FROM

    if(current_sequence_number_TO!=current_sequence_number_FROM):
        # current on_inspection file
        on_inspection_files = files[current_sequence_number_FROM:current_sequence_number_TO]
    else:
        on_inspection_files = files[current_sequence_number_TO]
    # BOUNDERIES FOR FILES: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # create initial on_inspection_dataset.csv with mfccs
    with open('on_inspection_dataset.csv','w') as file:
        # strLine = 'person, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13' # for default mfccs parameters
        strLine = 'person, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24' # for  bachlors mfccs parameters
        file.write(strLine)
        file.write('\n')
    # audio file on inspection:
    for on_inspection in on_inspection_files:
        # if on_inspection_files is already string:
        if isinstance(on_inspection_files, str):
            on_inspection = on_inspection_files

        label = 'on_inspection'

        # # for Dictors:>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # # let's convert .flac to .wav
        # on_inspection = AudioSegment.from_file(on_inspection, "flac")

        # # convert to .wav
        # on_inspection.export("on_inspection.wav", format="wav")
        # (rate,sig) = wav.read("on_inspection.wav")
        # # for Dictors <<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # for REAL VOICES: >>>>>>>>>>>>>>>>>>>>
        # read .wav file
        (rate,sig) = wav.read(on_inspection)
        # extract mfccs from demo.wav
        # for REAL VOICES <<<<<<<<<<<<<<<<<<<<


        # EXTRACT MFCCs FROM SIG, RATE:
        mfcc_feat = mfcc(sig,rate,winlen=0.094,nfft=FFT_LENGTH, numcep=numcep, lowfreq=lowfreq, highfreq=highfreq)

        #dump mfccs to .csv:
        # with open('on_inspection_dataset.csv','w+') as file:
        with open('on_inspection_dataset.csv','a') as file:
            for line in mfcc_feat:
                strLine = str(label) + ',' + ','.join(map(str, line))
                file.write(strLine)
                file.write('\n')


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

# Identification>>>>>>>>>>>>>>>>
# ==============================+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    # kneigbours:>>>
    # load the learned model
    clf = load('Classifier_model.joblib') 
    # let's predict 
    y_pred = clf.predict(X_inspect)
    # kneigbours <<<

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
    if second_max_elem == 0:
        first_max_second_max_relation = 9999
    # let's reverse from int to marks

    # load labelEncoder
    le = load('LabelEncoder.joblib') 
    # result_mark = le.inverse_transform([max_freq_item])
    result_mark = le.classes_[index_of_max_elem]

    IDENT_flag = 0

    # three state: ALLOW(2)/TRY MORE(1)/DENY(0):
    if(first_max_second_max_relation < minIdent):
        IDENT_flag=0 #DENY
    if(minIdent <= first_max_second_max_relation and first_max_second_max_relation < maxIdent):
        IDENT_flag=1 #TRY MORE
    if(first_max_second_max_relation>=maxIdent):
        IDENT_flag=2 #ALLOW

    print('relation:'+ str(first_max_second_max_relation))
# ==============================+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Identification<<<<<<<<<<<<<<<<

    # let's find all X_pred according to result_mark[0]
    demo_dataset = pd.read_csv('demo_dataset.csv') 
    result_individual = result_mark
    # dictors:
    pred = demo_dataset.loc[demo_dataset['person']==float(result_individual)]
    # # real voices:
    # pred = demo_dataset.loc[demo_dataset['person']==result_individual]
    # input data of predicted  individual
    # X_pred = pred.iloc[:, 1:14].values #for default mfccs parametrs
    X_pred = pred.iloc[:, 1:25].values #for mfccs parametrs from bachlor's work
    # output data of predicted  individual
    Y_pred = pred.iloc[:, 0].values
    # ==============================+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Varification
    return check_by_Verificator(X_inspect, Y_inspect, X_pred, Y_pred, result_individual, IDENT_flag)

# SINGLE-PERSON AUTHENTICATION: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
