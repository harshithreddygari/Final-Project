import json
import pandas as pd
import numpy as np
from csv import writer
import collections

import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

nan = float('nan')

def stress():
    counter = -1

    with open("stress.csv",'a') as f1:
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

    df_output = pd.read_csv("stress.csv",index_col=False)


    df_output = df_output.set_index('uid')


    df_entire = pd.concat([df_input,df_output],axis=1,join='outer')

    df_entire = df_entire.fillna(value=0)


    y= df_entire['mean_stress']

    list_columns=[]
    # for item in list_columns_temp:
    #     list_columns.append(item)
    # list_columns.remove('mean_stress')
    # list_columns.remove('std_stress')

    list_columns.append("std_call_duration")
    list_columns.append("proportion_running_walking")
    list_columns.append("mean_deadline")
    list_columns.append("mean_sleep")


    #
    df_entire.fillna(value=0)
    df_X = df_entire[list_columns]
    #
    #
    import statsmodels.api as sm
    from sklearn import linear_model,metrics,model_selection
    X_train,X_test,Y_train,Y_test = model_selection.train_test_split(df_X,y,train_size=0.8)
    model = sm.OLS(Y_train,X_train)
    results = model.fit()
    print(results.summary())
    print("Actual vs Prediction")
    d = pd.DataFrame()
    d["Actual"] = Y_test
    d["Predicted"] = results.predict(X_test)
    d["Error"] = d["Actual"] - d["Predicted"]
    print(d)



if __name__ == '__main__':

    # socialEngagement()
    # mobility()
    # activity()
    # phone_activity()
    # deadlines()
    # sleep()
    # stress()
    ip1 = pd.read_csv('mobility.csv', index_col=False)
    ip2 = pd.read_csv('socialEngagement.csv', index_col=False)
    ip3 = pd.read_csv('phone_activity.csv', index_col=False)
    ip4 = pd.read_csv('physical_activity.csv', index_col=False)
    ip5 = pd.read_csv('deadline_activity.csv', index_col=False)
    ip6 = pd.read_csv('sleep.csv', index_col=False)


    ip1 = ip1.set_index('uid')
    ip2 = ip2.set_index('uid')
    ip3 = ip3.set_index('uid')
    ip4 = ip4.set_index('uid')
    ip5 = ip5.set_index('uid')
    ip6 = ip6.set_index('uid')



    ip = pd.concat([ip1,ip2,ip3,ip4,ip5,ip6],axis=1,join='outer')
    ip = ip.fillna(value=0)
    regression(ip)