import numpy as np
import json
import pandas as pd
from csv import writer
from numpy import *
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import perceptron
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

resultCoeff=[]
nan = float('nan')
def allClassifiers():
    names = ["Decision Trees", "Perceptron", "Neural Net", "Deep Learning", "SVM", "Naive Bayes", "Logostic Regression",
             "k-NN", "Bagging", "Random Forests",
             "Adaboost", "Gradient Boosting"]
    classifiers = [
        DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                               max_features='log2', max_leaf_nodes=None,
                               min_impurity_split=1e-07, min_samples_leaf=1,
                               min_samples_split=8, min_weight_fraction_leaf=0.1,
                               presort=False, random_state=None, splitter='best'),
        perceptron.Perceptron(n_iter=1000, verbose=0, random_state=10, fit_intercept=True, eta0=0.008,
                              alpha=0.000223, max_iter=1000),
        MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
                      beta_1=0.9, beta_2=0.999, early_stopping=False,
                      epsilon=1e-08, hidden_layer_sizes=(20, 20, 10), learning_rate='constant',
                      learning_rate_init=0.001, max_iter=200, momentum=0.9,
                      nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
                      solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
                      warm_start=False),
        MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
                      beta_1=0.9, beta_2=0.999, early_stopping=False,
                      epsilon=1e-08, hidden_layer_sizes=(10, 10, 10, 20, 10, 10, 10), learning_rate='adaptive',
                      learning_rate_init=0.001, max_iter=200, momentum=0.9,
                      nesterovs_momentum=True, power_t=0.5, random_state=4, shuffle=True,
                      solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
                      warm_start=False),
        SVC(C=1.0, cache_size=300, class_weight=None, coef0=0.0,
            decision_function_shape='ovo', degree=7, gamma='auto', kernel='linear',
            max_iter=-1, probability=True, random_state=None, shrinking=True,
            tol=0.001, verbose=False),
        GaussianNB(),
        LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1e5, fit_intercept=True,
                           intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear',
                           max_iter=10000, multi_class='ovr', verbose=10, warm_start=True, n_jobs=1),
        KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                             metric_params=None, n_jobs=1, n_neighbors=30, p=2,
                             weights='distance'),
        BaggingClassifier(base_estimator=None, n_estimators=50, max_samples=1.0, max_features=1.0,
                          bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False,
                          n_jobs=1, random_state=None, verbose=0),
        RandomForestClassifier(n_estimators=200, criterion='entropy', max_depth=None, min_samples_split=2,
                               min_samples_leaf=2, min_weight_fraction_leaf=0.0, max_features=2,
                               max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
                               bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0,
                               warm_start=False, class_weight=None),
        AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200, learning_rate=1.0,
                           random_state=None),
        GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=200, subsample=1.0,
                                   criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1,
                                   min_weight_fraction_leaf=0.0,
                                   max_depth=2, min_impurity_decrease=0.0, min_impurity_split=None, init=None,
                                   random_state=None,
                                   max_features=2, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')]
    df = pd.read_csv(
        "completedataset_fixed.csv", header=1,
        names=["total_location", "unique_location", "total_calls", "total_conversations", "mean_call_duration",
               "std_call_duration", "mean_duration_dark", "std_dark", "mean_lock_duration", "std_lock",
               "most_freq_activity", "proportion_running_walking", "mean_deadline", "std_deadline", "mean_sleep",
               "std_sleep", "mean_stress", "std_stress", "labels"])
    array = df.values
    df.apply(pd.to_numeric, errors='ignore')
    df.fillna(method='ffill')
    X = array[:, :-1]
    where_are_NaNs = isnan(X)
    X[where_are_NaNs] = 0
    Y = df['labels']
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    folds = 5
    kf = KFold(n_splits=folds)
    classifier_accuracy_dict = dict()
    classifier_roc_score_dict = dict()
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        for name, clf in zip(names, classifiers):
            clf.fit(X_train, Y_train)
            predictions = clf.predict(X_test)
            accuracy = accuracy_score(Y_test, predictions)
            roc_score = roc_auc_score(Y_test, predictions)
            if classifier_accuracy_dict.get(name):
                classifier_accuracy_dict[name] += accuracy
                classifier_roc_score_dict[name] += roc_score
            else:
                classifier_accuracy_dict[name] = accuracy
                classifier_roc_score_dict[name] = roc_score
    for name in classifier_accuracy_dict.keys():
        print("Clasifier name: ", name, ", Avg Accuracy: ", classifier_accuracy_dict[name] / folds, ", Avg ROC Score: ",
              classifier_roc_score_dict[name] / folds)
def svm():
    df = pd.read_csv(
        "completedataset_fixed.csv", header=1,
        names=["total_location", "unique_location", "total_calls", "total_conversations", "mean_call_duration",
               "std_call_duration", "mean_duration_dark", "std_dark", "mean_lock_duration", "std_lock",
               "most_freq_activity", "proportion_running_walking", "mean_deadline", "std_deadline", "mean_sleep",
               "std_sleep", "mean_stress", "std_stress", "labels"])
    df.apply(pd.to_numeric, errors='ignore')
    df.fillna(method='ffill')
    array = df.values
    X = array[:, :-1]
    where_are_NaNs = isnan(X)
    X[where_are_NaNs] = 0
    Y = df['labels']
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    folds = 10
    kf = KFold(n_splits=folds)
    sum_accuracy = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                  decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
                  max_iter=-1, probability=False, random_state=None, shrinking=True,
                  tol=0.001, verbose=False)
        clf.fit(X_train, Y_train)
        predictions = clf.predict(X_test)
        print(confusion_matrix(Y_test, predictions))
        print(classification_report(Y_test, predictions))
        sum_accuracy += accuracy_score(Y_test, predictions)
    print("Avg Accuracy: ", sum_accuracy / folds)

def logisticRegression():

    df = pd.read_csv(
        "completedataset_fixed.csv", header=1,
        names=["total_location", "unique_location", "total_calls", "total_conversations", "mean_call_duration",
               "std_call_duration", "mean_duration_dark", "std_dark", "mean_lock_duration", "std_lock",
               "most_freq_activity", "proportion_running_walking", "mean_deadline", "std_deadline", "mean_sleep",
               "std_sleep", "mean_stress", "std_stress", "labels"])
    df.apply(pd.to_numeric, errors='ignore')
    df.fillna(method='ffill')
    df.describe()
    array = df.values
    X = array[:, :-1]

    where_are_NaNs = isnan(X)
    X[where_are_NaNs] = 0
    Y = df['labels']
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    folds = 10
    kf = KFold(n_splits=folds)
    sum_accuracy = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        clf = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1e3, fit_intercept=True,
                                 intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs',
                                 max_iter=100, multi_class='ovr', verbose=10, warm_start=True, n_jobs=1)
        clf.fit(X_train, Y_train)
        predictions = clf.predict(X_test)
        print(confusion_matrix(Y_test, predictions))
        print(classification_report(Y_test, predictions))
        sum_accuracy += accuracy_score(Y_test, predictions)
    print("Avg Accuracy: ", sum_accuracy / folds)

def randomforest():

    df = pd.read_csv(
        "completedataset_fixed.csv", header=1,
        names=["total_location", "unique_location", "total_calls", "total_conversations", "mean_call_duration",
               "std_call_duration", "mean_duration_dark", "std_dark", "mean_lock_duration", "std_lock",
               "most_freq_activity", "proportion_running_walking", "mean_deadline", "std_deadline", "mean_sleep",
               "std_sleep", "mean_stress", "std_stress", "labels"])

    df.apply(pd.to_numeric, errors='ignore')
    df.fillna(method='ffill')
    df.describe()
    array = df.values
    X = array[:, :-1]
    where_are_NaNs = isnan(X)
    X[where_are_NaNs] = 0

    #   print X
    Y = df['labels']
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    folds = 10
    kf = KFold(n_splits=folds)
    sum_accuracy = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        mlp = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2,
                                     min_samples_leaf=2, min_weight_fraction_leaf=0.0, max_features='auto',
                                     max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
                                     bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0,
                                     warm_start=False, class_weight=None)
        mlp.fit(X_train, Y_train)
        predictions = mlp.predict(X_test)
        print(confusion_matrix(Y_test, predictions))
        print(classification_report(Y_test, predictions))
        sum_accuracy += accuracy_score(Y_test, predictions)
    print("Avg Accuracy: ", sum_accuracy / folds)



def stress():
    counter = -1

    with open("stress.csv",'w') as f1:
            csvWriter = writer(f1)
            csvWriter.writerow(['uid','mean_stress','std_stress'])

    while counter<=60:
        counter+=1
        temp_list = []

        if counter < 10:
            user_id = 'u0' + str(counter)
        else:
            user_id = 'u' + str(counter)

        temp_list.append(user_id)

        try:
            data = json.load(open('StudentLife_AssignmentData/SensingData/Stress/Stress_' + user_id + '.json'))
            df = pd.DataFrame.from_dict(data)
        except:
            continue
        df = df.fillna(value=0)
        try:
            l=df["level"].tolist()
            l = pd.to_numeric(l)
        except:
            continue

        temp_list.append(np.mean(l))
        temp_list.append(np.std(l))

        with open("stress.csv",'a') as f1:
            csvWriter = writer(f1)
            csvWriter.writerow(temp_list)

def sleep():
    counter = -1

    with open("sleep.csv",'a') as f1:
            csvWriter = writer(f1)
            csvWriter.writerow(['uid','mean_sleep','std_sleep'])

    while counter<=60:
        counter+=1
        temp_list = []

        if counter < 10:
            user_id = 'u0' + str(counter)
        else:
            user_id = 'u' + str(counter)

        temp_list.append(user_id)

        try:
            data = json.load(open('StudentLife_AssignmentData/SensingData/Sleep/Sleep_' + user_id + '.json'))
            df = pd.DataFrame.from_dict(data)
        except:
            continue
        df = df.fillna(value=0)
        l=df["hour"].tolist()
        l = pd.to_numeric(l)

        temp_list.append(np.mean(l))
        temp_list.append(np.std(l))

        with open("sleep.csv",'a') as f1:
            csvWriter = writer(f1)
            csvWriter.writerow(temp_list)


def deadlines():

    deadline = pd.read_csv("StudentLife_AssignmentData/SensingData/education/deadlines.csv", index_col=False)
    deadline["mean_deadline"] = deadline[1:].mean(axis=1)
    deadline["std_deadline"] = deadline[1:].std(axis=1)
    deadline = deadline.fillna(value=0)
    with open("deadline_activity.csv", 'a') as f1:
        csvWriter = writer(f1)
        csvWriter.writerow(['uid', 'mean_deadline', 'std_deadline'])

        for index,item in deadline.iterrows():
            temp = []
            temp.append(item['uid'])
            temp.append(item['mean_deadline'])
            temp.append(item['std_deadline'])
            csvWriter.writerow(temp)

def mentalWellBeing():

    df_phq = pd.read_csv("StudentLife_AssignmentData/Surveys/PHQ-9.csv",index_col=False)
    list_columns = []
    list_columns = (df_phq.columns.values).tolist()
    list_columns.remove('uid')
    list_columns.remove('type')


    #raw_input()
    with open("file4.csv", 'w') as f1:
        csvWriter = writer(f1)
        header_list = []
        header_list.append('uid')
        for listItem in list_columns:
            header_list.append(listItem)
        csvWriter.writerow(['uid','phq-9'])
        for index, item in df_phq.iterrows():
            temp_uid = item['uid']
            temp_list = []

            temp_list.append(item['uid'])
            for index2, item2 in df_phq.iterrows():
                if item2['uid'] == temp_uid and item2['type'] == 'post' and item['type'] == 'pre':
                    for columnValue in list_columns:

                        temp = 0
                        dict = {
                            'not at all':0,
                            'several days':1,
                            'more than half the days':2,
                            'nearly every day':3,
                            'not difficult at all':0,
                            'somewhat difficult':1,
                            'very difficult':2,
                            'extremely difficult':3

                        }
                        try:
                            value1 = dict[item[columnValue].lower()]
                        except:
                            value1=0
                        #print value1
                        #raw_input()
                        if math.isnan(value1):
                            #print "here"
                            #raw_input('nan')
                            value1 = 0

                        try:
                            value2 = dict[item2[columnValue].lower()]
                        except:
                            value2=0
                        temp = (value1 + value2)/float(2)
                        temp_list.append(temp)

                    csvWriter.writerow([temp_list[0],sum(temp_list[1:])])
                    continue


    df1 = pd.read_csv("file4.csv",index_col=False)


    df1 = df1.set_index('uid')


    result = pd.concat([df1],axis=1,join='inner')

    result.to_csv("output_mentalWellBeing.csv")

def socialEngagement():
    counter = -1

    with open("socialEngagement.csv",'a') as f1:
            csvWriter = writer(f1)
            csvWriter.writerow(['uid','total_calls','total_conversations','mean_call_duration','std_call_duration'])

    while counter<=60:
        counter+=1
        temp_list = []

        if counter < 10:
            user_id = 'u0' + str(counter)
        else:
            user_id = 'u' + str(counter)

        temp_list.append(user_id)

        total_calls = 0
        try:
            df = pd.read_csv('StudentLife_AssignmentData/SensingData/CallLog/call_log_' + user_id + '.csv',index_col=False)
        except:
            continue
        total_calls = len(df)
        temp_list.append(total_calls)

        try:
            df = pd.read_csv('StudentLife_AssignmentData/SensingData/Conversations/conversation_' + user_id + '.csv',index_col=False)
        except:
            continue
        total_conversations = 0
        total_conversations=len(df)
        temp_list.append(total_conversations)

        list_timeDiff = []
        print df.columns.values
        for index,item in df.iterrows():
            list_timeDiff.append(item[' end_timestamp']-item['start_timestamp'])
        mean_call_duation = np.mean(list_timeDiff)
        std_call_duration = np.std(list_timeDiff)

        temp_list.append(mean_call_duation)
        temp_list.append(std_call_duration)

        with open("socialEngagement.csv",'a') as f1:
            csvWriter = writer(f1)
            csvWriter.writerow(temp_list)

def mobility():
    counter = -1

    with open("mobility.csv",'a') as f1:
            csvWriter = writer(f1)
            csvWriter.writerow(['uid','total_location','unique_location'])

    while counter<=60:
        counter+=1
        temp_list = []

        if counter < 10:
            user_id = 'u0' + str(counter)
        else:
            user_id = 'u' + str(counter)

        temp_list.append(user_id)

        total_locations = 0
        try:
            df = pd.read_csv('StudentLife_AssignmentData/SensingData/Wifi_Location/wifi_location_' + user_id + '.csv',index_col=False)
        except:
            continue
        #print df
        total_locations=len(df)
        #print total_locations
        temp_list.append(total_locations)
        list_locations = []

        list_locations=df['location'].tolist()
        # for index, item in df.iterrows():
        #     #print type(item['location'])
        #     #print item
        #     #raw_input()
        #     list_locations.append(item['location'])

        set_locations = set(list_locations)

        #print len(list_locations)

        # for item in list_locations:
        #     print item
            #raw_input()
        #print len(set_locations)
        #raw_input()

        temp_list.append(len(set_locations))
        #raw_input()
        with open("mobility.csv",'a') as f1:
            csvWriter = writer(f1)
            csvWriter.writerow(temp_list)


def activity():
    counter = -1

    with open("physical_activity.csv",'a') as f1:
            csvWriter = writer(f1)
            csvWriter.writerow(['uid','most_freq_activity','proportion_running_walking'])

    while counter<=60:
        counter+=1
        temp_list = []

        if counter < 10:
            user_id = 'u0' + str(counter)
        else:
            user_id = 'u' + str(counter)

        temp_list.append(user_id)


        try:
            df = pd.read_csv('StudentLife_AssignmentData/SensingData/PhysicalActivity/activity_' + user_id + '.csv',index_col=False)
        except:
            continue

        list_activities = df[' activity inference'].tolist()
        print len(list_activities)
        count_activity = collections.Counter(list_activities)

        most_freq = count_activity.most_common(1)
        temp_list.append(most_freq[0][0])

        average = (count_activity[1] + count_activity[2])/float(len(list_activities))
        temp_list.append(average)
        print count_activity
        #print count_activity[0]
        #raw_input()

        # for index, item in df.iterrows():
        #     list_activities.append(item[' activity inference'])
        with open("physical_activity.csv",'a') as f1:
            csvWriter = writer(f1)
            csvWriter.writerow(temp_list)
def phone_activity():
    counter = -1

    with open("phone_activity.csv",'a') as f1:
            csvWriter = writer(f1)
            csvWriter.writerow(['uid','mean_duration_dark','std_dark','mean_lock_duration','std_lock'])

    while counter<=60:
        counter+=1
        temp_list = []

        if counter < 10:
            user_id = 'u0' + str(counter)
        else:
            user_id = 'u' + str(counter)

        temp_list.append(user_id)


        try:
            df = pd.read_csv('StudentLife_AssignmentData/SensingData/PhoneLight/dark_' + user_id + '.csv',index_col=False)
        except:
            continue

        list_dark_duration = []
        for index,item in df.iterrows():
            list_dark_duration.append(item['end']-item['start'])
        temp_list.append(np.mean(list_dark_duration))
        temp_list.append(np.std(list_dark_duration))

        try:
            df = pd.read_csv('StudentLife_AssignmentData/SensingData/PhoneLock/phonelock_' + user_id + '.csv',index_col=False)
        except:
            continue

        list_lock_duration = []
        for index,item in df.iterrows():
            list_lock_duration.append(item['end']-item['start'])
        temp_list.append(np.mean(list_lock_duration))
        temp_list.append(np.std(list_lock_duration))

        with open("phone_activity.csv",'a') as f1:
            csvWriter = writer(f1)
            csvWriter.writerow(temp_list)

def regression(df_input):
    global resultcoeff

    #df_input=pd.read_csv(trainingFile,index_col=False)
    df_output = pd.read_csv("stress.csv",index_col=False)

    #df_input = df_input.set_index('uid')
    df_output = df_output.set_index('uid')

    #print(df_input)
    df_entire = pd.concat([df_input,df_output],axis=1,join='outer')

    df_entire = df_entire.fillna(value=0)
    #print df_entire.isnull().values.any()
    #print df_entire.columns.values
    # df_entire.to_csv("Final.csv",sep=",")
    y= df_entire['mean_stress']


    list_columns_temp = (df_entire.columns.values)
    list_columns=[]
    for item in list_columns_temp:
        list_columns.append(item)
    list_columns.remove('mean_stress')
    list_columns.remove('std_stress')

    df_X = df_entire[list_columns]

    import statsmodels.api as sm

    model = sm.OLS(y[0:35],df_X[0:35])
    results = model.fit()
    resultcoeff = results.params
#    print(resultcoeff)

    print(results.summary())

    print(results.predict(df_X[36:]))


def feature_derivative(errors, feature):
    # Compute the dot product of errors and feature
    derivative = np.dot(errors, feature)
        # Return the derivative
    return derivative
def compute_log_likelihood(feature_matrix,label, coefficients):
    indicator = (label==+1)
    scores = np.dot(feature_matrix, coefficients)
    lp = np.sum((indicator-1)*scores - np.log(1. + np.exp(-scores)))
    return lp

from math import sqrt
def logistic_regression(feature_matrix, label, initial_coefficients, step_size, max_iter):
    coefficients = np.array(initial_coefficients) # make sure it's a numpy array
    for itr in xrange(max_iter):
        # Predict P(y_i = +1|x_1,w) using your predict_probability() function

        predictions = predict_probability(feature_matrix, coefficients)

        # Compute indicator value for (y_i = +1)
        indicator = (label==+1)

        # Compute the errors as indicator - predictions
        errors = indicator - predictions

        for j in xrange(len(coefficients)): # loop over each coefficient
            # Recall that feature_matrix[:,j] is the feature column associated with coefficients[j]
            # compute the derivative for coefficients[j]. Save it in a variable called derivative
            # YOUR CODE HERE
            derivative = np.dot(errors, feature_matrix[:,j])

            # add the step size times the derivative to the current coefficient

            coefficients[j] = step_size * derivative

        # Checking whether log likelihood is increasing
        if itr <= 15 or (itr <= 100 and itr % 10 == 0) or (itr <= 1000 and itr % 100 == 0) \
        or (itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0:
            lp = compute_log_likelihood(feature_matrix, label, coefficients)
            print 'iteration %*d: log likelihood of observed labels = %.8f' % \
                (int(np.ceil(np.log10(max_iter))), itr, lp)
    return coefficients
def stresswithlabels():
    counter = -1
    list_mean_append =0
    with open("stress_and_labels.csv",'w') as f1:
            csvWriter = writer(f1)
            csvWriter.writerow(['uid','mean_stress','std_stress','labels'])

    while counter<=60:
        counter+=1
        temp_list = []

        if counter < 10:
            user_id = 'u0' + str(counter)
        else:
            user_id = 'u' + str(counter)

        temp_list.append(user_id)

        try:
            data = json.load(open('StudentLife_AssignmentData/SensingData/Stress/Stress_' + user_id + '.json'))
            df = pd.DataFrame.from_dict(data)
        except:
            continue
        df = df.fillna(value=0)
        try:
            l=df["level"].tolist()
            l = pd.to_numeric(l)

        except:
            continue
        list_mean=np.mean(l)
        list_stddev= np.std(l)


        if(nan):
            list_mean_append=0
        if list_mean>1.25 :
            list_mean_append=1
        else:
            list_mean_append=0

        temp_list.append(np.mean(l))
        temp_list.append(np.std(l))
        temp_list.append(list_mean_append)

        with open("stress_and_labels.csv",'a') as f1:
            csvWriter = writer(f1)
            csvWriter.writerow(temp_list)



if __name__ == '__main__':
    # mentalWellBeing()
    # socialEngagement()
    # mobility()
    # activity()
    # phone_activity()
    # deadlines()
    # sleep()
    stress()
    stresswithlabels()

    ip1 = pd.read_csv('mobility.csv', index_col=False)
    ip2 = pd.read_csv('socialEngagement.csv', index_col=False)
    ip3 = pd.read_csv('phone_activity.csv', index_col=False)
    ip4 = pd.read_csv('physical_activity.csv', index_col=False)
    ip5 = pd.read_csv('deadline_activity.csv', index_col=False)
    ip6 = pd.read_csv('sleep.csv', index_col=False)
    ip7 = pd.read_csv('stress_and_labels.csv', index_col=False)

    ip1 = ip1.set_index('uid')
    ip2 = ip2.set_index('uid')
    ip3 = ip3.set_index('uid')
    ip4 = ip4.set_index('uid')
    ip5 = ip5.set_index('uid')
    ip6 = ip6.set_index('uid')
    ip7 = ip7.set_index('uid')
#    df = pd.read_csv('stress.csv', index_col=False)

    #list_columnvalues=[]
   # list_newcolumn=[]
  #  for i in df["mean_stress"]:

     #   if(i>1.5):
      #      list_newcolumn.append("1")
       # else:
        #    list_newcolumn.append("0")
    #with open('stress.csv','a') as f1:
    #    csvWriter=writer(f1)
     #   csvWriter.writerow(list_newcolumn)


    #df.to_csv("stress_with_labels.csv",index=False)



    ip = pd.concat([ip1,ip2,ip3,ip4,ip5,ip6],axis=1,join='outer')
    ip = ip.fillna(value=0)
    ip.to_csv("iputdatasetregress.csv")
    ip_class = pd.concat([ip1,ip2,ip3,ip4,ip5,ip6,ip7],axis=1,join='outer')
    ip_class.fillna(value=0)
    ip_class.to_csv("completedataset.csv",index='uid')
    in_file = "completedataset.csv"
    out_file = "completedataset_fixed.csv"

 #   row_reader = csv.reader(open(in_file, "rb"))
 #   row_writer = csv.writer(open(out_file, "wb"))

#    first_row = row_reader.next()
#    row_writer.writerow(first_row)
 #   for row in row_reader:
  #      new_row = [val if val else "0" for val in row] + (["0"] * (len(first_row) - len(row)))
   #     print row, "->", new_row
    #    row_writer.writerow(new_row)


   # global label_array
    #column_names = ["total_location", "unique_location", "total_calls", "total_conversations", "mean_call_duration",
                #    "std_call_duration", "mean_duration_dark", "std_dark", "mean_lock_duration", "std_lock",
                 #   "most_freq_activity", "proportion_running_walking", "mean_deadline", "std_deadline", "mean_sleep",
                  #  "std_sleep","mean_stress","std_stress","labels"]

#    regression(ip)
 #   (feature_matrix,label_array) = getnumpyData(ip_class, column_names, ip_class['labels'])

  #  features_matrix = feature_matrix
  #  label = label_array
  #  initial_coefficients = np.zeros(194)
  #  step_size = 1e-7
  #  max_iter = 301
 #   variable_coefficients = logistic_regression(features_matrix, label, initial_coefficients, step_size, max_iter)
   # randomforest()
  #  logisticRegression()
   # svm()
    allClassifiers()


