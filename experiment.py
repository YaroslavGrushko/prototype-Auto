from experiment_helper.header import *
# authentification
from experiment_helper.helper import authentification
from experiment_helper.helper import check_by_Verificator

if __name__ == '__main__':

    if not os.path.exists(src_results_folder):
            os.makedirs(src_results_folder)

    # let's initialize FFT_LENGTH with 128 default value
    # this value will be increased after time
    
    # FFT_LENGTH = 65536
    FFT_LENGTH = 8192
    while FFT_LENGTH < 8193:
        
        current_demo_dataset_results = str(src_results_folder)+"demo_dataset_results"+str(FFT_LENGTH)+".csv"
        with open(current_demo_dataset_results,'w') as file:
            my_string = "TIME_TO_SPLIT,Allow_true,Allow_false,all_count"
            file.write(my_string)
            file.write('\n')

        # TIME_TO_SPLIT = 3.0   
        # time to split audio file (in seconds)
        for TIME_TO_SPLIT in np.arange(4.5, 5.0, 0.5):
            # counter  of inner experiment
            inner_expariment_counter = 0

            # let's import text_into_phrases.py module >>>>>>>
            # let's split voices:
            split_all_files(PERSONS_DIRECTORY_FROM, dataset_directory, TIME_TO_SPLIT)
            # let's import text_into_phrases.py module <<<<<<<


            # all persons in main dirname
            all_persons = os.listdir(dataset_directory)
            min_len = 999999
            # let's find min number of files in a person folder >>>
            for person_dir in all_persons:
                full_person_dir = os.path.join(dataset_directory, person_dir)
                files = os.listdir(full_person_dir)
                if len(files)<min_len:
                    min_len = len(files)
            # <<<

            all_count = 0
            true_allow_count = 0
            false_allow_count = 0
            # FOREACH FILES IN PERSONS FOLDER: (FROM START AND FROM END)>>>>>>>>>>>>>>>>>
            # let's foreach number_to_learn_FROM (write to db) and number_from_end_FROM (authentication)
            for number_to_learn_FROM in range(0, min_len, 3):
                for number_from_end_FROM in range(0, min_len):
                    # foreach until write_to_db and authentication intersect
                    if number_to_learn_FROM < min_len - 1 - number_from_end_FROM:
                        # let's make maximum boarder for inner experiments count
                        if all_count>MAX_inner_experiment_count:
                            break

                        # ABOUT DATASET #2 (WRITE_TO_DB) >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                        # number of records to learn:
                        number_to_learn = 3
                        # sequence number of record to learn TO
                        number_to_learn_TO = number_to_learn_FROM+number_to_learn
                        # ABOUT DATASET <<<<<<<<<<<<<<< <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


                        # ABOUT DATASET #3 (AUTHENTICATION) >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                        # number from of the files for current person
                        number_of_items = 1
                        # squence number from end of the file for current person TO
                        number_from_end_TO = number_from_end_FROM+number_of_items-1
                        # ABOUT DATASET <<<<<<<<<<<<<<< <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

                        # WRITE_TO_DB >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                        # create dataset.csv with mfccs header
                        with open('demo_dataset.csv','w') as file:
                            # strLine = 'person, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13'
                            strLine = 'person, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24'
                            file.write(strLine)
                            file.write('\n')

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
                                # # for DICTORS: >>>>>>>>>>>>>>>>>>>>>>>>>
                                #     # let's convert .flac to .wav
                                # demo = AudioSegment.from_file(demo, "flac")

                                # # convert to .wav
                                # demo.export("demo.wav", format="wav")
                                # (rate,sig) = wav.read("demo.wav")
                                # # for DICTORS <<<<<<<<<<<<<<<<<<<<<<<<<

                                # for REAL VOICES: >>>>>>>>>>>>>>>>>>>>
                                # read .wav file
                                (rate,sig) = wav.read(demo)
                                # extract mfccs from demo.wav
                                # for REAL VOICES <<<<<<<<<<<<<<<<<<<<
                                
                                # EXTRACT MFCCs FROM SIG, RATE:
                                mfcc_feat = mfcc(sig,rate,winlen=0.094,nfft=FFT_LENGTH, numcep=numcep, lowfreq=lowfreq, highfreq=highfreq) #bachlor parameter

                                #let's dump mfccs string to .csv:
                                with open('demo_dataset.csv','a') as file:
                                    for line in mfcc_feat:
                                        strLine = str(LABEL) + ',' + ','.join(map(str, line))
                                        file.write(strLine)
                                        file.write('\n')
                            # message to terminal after every person is recorded to DB-file
                            print("mfccs of "+str(person_dir)+" was recorded to .csv")
                            # <<< foreach persons to .csv

                        # WRITE_TO_DB <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

                        # LEARNING: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                        # read DB-file with mfccs:
                        # header = None if dataset has no header
                        dataset =  pd.read_csv('demo_dataset.csv',sep=',')

                        # let's encode catigorial data to numbers
                        dataset_encoded = dataset.copy() 
                        # creating labelEncoder
                        le = preprocessing.LabelEncoder()
                        # Converting string labels into numbers.
                        dataset_encoded['person']=le.fit_transform(dataset['person'].astype(str))
                        # # let's override
                        # mylableEncoder = open('LabelEncoder.joblib','wb')
                        # let's save LabelEncoder on disk
                        dump(le, 'LabelEncoder.joblib')

                        # make 2 datasets X - for input data and Y- for output
                        #  input data
                        # X = dataset_encoded.iloc[:, 1:14].values  #for default mfccs parameters
                        X = dataset_encoded.iloc[:, 1:25].values # for bachloar mfccs parametrs

                        #  output data
                        Y = dataset_encoded.iloc[:, 0].values

                        # CLASSIFIER>>>>>>>>>>>>>>>>>>>>>>:
                        #  KNeighbours:
                        clf = KNeighborsClassifier(n_neighbors=3)
                        # # mlp:
                        # clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
                        # # svm:
                        # clf = SVC(gamma='auto')
                        # # DecisionTreeClassifier:
                        # clf =  DecisionTreeClassifier(max_depth=5)
                        # # Naive bayes:
                        # clf = GaussianNB()
                        # # AdaBoost:
                        # clf = AdaBoostClassifier()
                        # # random forest classifier
                        # clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
                        # << CLASSIFIER <<<<<<<<<<<<<<<<<

                        # let's train:
                        clf.fit(X, Y) 

                        ## let's override
                        # myClassifier_model = open('Classifier_model.joblib','wb')
                        # let's save model on disk
                        dump(clf, 'Classifier_model.joblib')
                        # model was trained
                        print("model was trained")
                        # LEARNING: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

                        # AUTHENTICATION >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

                        true_count = 0
                        false_count = 0
                        try_more_count = 0
                        # read persons
                        person_dirs = os.listdir(dataset_directory)
                        # by default - zeros
                        persons_results = np.zeros(len(person_dirs))
                        # creaate results pandas
                        persons_df = pd.DataFrame({'names':person_dirs,'result':persons_results})
                        persons_df_ident = pd.DataFrame({'names':person_dirs,'result':persons_results})

                        # create copy
                        persons_results_df = persons_df.copy()
                        persons_df_ident_result = persons_df_ident.copy()


                        # all persons that are given to learn:
                        
                        # all persons in main dirname
                        all_persons = os.listdir(dataset_directory)
                        # # persons that we select to write to Db (authentication parameter)
                        # persons_to_learn = all_persons[:number_of_persons_to_learn] 

                        # foreach all persons (folders)
                        for person_dir in person_dirs:
                            # full_person_directory
                            full_person_dir = os.path.join(dataset_directory, person_dir)      
                            # let's get prediction result for current person
                            person_name_pred, action = authentification(full_person_dir,number_from_end_TO,number_from_end_FROM, FFT_LENGTH)

                            person_name_pred = str(person_name_pred)
                            # we must each every person in dir in last batch
                            if all_count>MAX_inner_experiment_count and (all_count%len(person_dirs)==0):
                                break
                            all_count +=1
                            # let's write statistics about current person to general pandas:
                            # if identification is true and action is 'allow'
                            if person_name_pred == person_dir and action == 'ALLOW':
                                # persons_results_df.loc[persons_results_df.names == person_dir]=person_dir, 2
                                # true_count+=1
                                true_allow_count+=1
                                print("allow_true")
                                
                            # # if identification is false and action is 'deny'
                            # elif person_name_pred != person_dir and action == 'DENY':
                            #     if not (person_dir in persons_to_learn):
                            #         persons_results_df.loc[persons_results_df.names == person_dir]=person_dir, 1
                            #         true_count+=1
                            #         print("true")
                            #     else:
                            #         false_count+=1
                            #         print("false") 
                            # # if action is 'try more'
                            # elif action == 'TRY MORE':
                            #     try_more_count+=1

                            # else:
                            #     false_count+=1
                            #     print("false")
                            if not (person_name_pred == person_dir) and action == 'ALLOW':
                                false_allow_count+=1
                                print("allow_false")
                            # # if identification 'false'
                            # if person_name_pred == person_dir:
                            #     persons_df_ident_result.loc[persons_df_ident_result.names == person_dir]=person_dir, 1

                            
                        # print('persons_results_df len: '+str(len(persons_results_df)))
                        # statistics = persons_results_df.groupby("result")
                        # print('results (40 persons + 3 train phrase, 1 test phrase + 1 time test): ')
                        # for key, item in statistics:
                        #     print ('key: '+ str(key)+' count: '+str(len(statistics.get_group(key))), "\n")
                        
                        # # true/false statistics
                        # print("true count: "+str(true_count))
                        # print("false count: "+str(false_count), "\n")
                        # print("try_more count: "+str(try_more_count), "\n")


                        # # identification statistics:
                        # print('identification statistics')
                        # ident_statistics = persons_df_ident_result.groupby("result")
                        # for key, item in ident_statistics:
                        #     print ('key: '+ str(key)+' count: '+str(len(ident_statistics.get_group(key))), "\n")
        
            with open(current_demo_dataset_results,'a') as file:
                strLine = str(TIME_TO_SPLIT)+","+str(true_allow_count)+","+str(false_allow_count)+","+str(all_count)
                file.write(strLine)
                file.write('\n')
                    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Authentication 
        FFT_LENGTH=FFT_LENGTH*2
        # FOREACH FILES IN PERSONS FOLDER: (FROM START AND FROM END)<<<<<<<<<<<<<<<<<

    
