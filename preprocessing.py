import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import math
from scipy import io
import random
from datetime import datetime, date
import math

def dataset_to_dataframe(traindatax,traindatay ,  featurenames,yfeature):
    dfx = pd.DataFrame(traindatax,  columns=featurenames)

    if(yfeature==1):
        dfx ["LABEL"] = traindatay
    return dfx

def parsefile(filename,yfeature):
    fp = open(filename,'r')
    datax =[]
    datay =[]
    for line in fp:
        data=line.strip().split('\t')
        if(yfeature==1):
            datax.append(data[:-1])
            datay.append(data[-1])
        else:
            datax.append(data)
    return datax,datay

def parsefile_val_and_test(filename,yfeature):
    fp = open(filename,'r')
    datax =[]
    datay =[]
    for line in fp:
        data=line.strip().split(',')
        if(yfeature==1):
            datax.append(data[:-1])
            datay.append(data[-1])
        else:
            datax.append(data)
    return datax,datay

def build_data(data,dfdata_question_info,dfdata_user_info):
    result = pd.merge(data, dfdata_question_info, how='left', on=['QID'])
    result1 = pd.merge(result, dfdata_user_info, how='left', on=['EID'])
    #res = result1.drop(["CHAR_ID_SEQ","USER_CHAR_ID_SEQ"],axis = 1)
    # print res.columns
    #
    # print "data"
    # for col in data.columns:
    #     print data[col][0],
    #
    # print "result"
    # for col in result.columns:
    #     print result[col][0],
    #
    # print "result1"
    # for col in result1.columns:
    #     print result1[col][0],

    return result1



if __name__ == "__main__":
    filenameinfotrain = "data/invited_info_train.txt"
    datax,datay = parsefile(filenameinfotrain,1)
    arrayoffeaturenames =["QID","EID"]
    dfdata_qid_eid_class = dataset_to_dataframe(datax,datay,arrayoffeaturenames,1)

    filenameinfotrain = "data/question_info.txt"
    datax,datay = parsefile(filenameinfotrain,0)
    arrayoffeaturenames =["QID","QTAG","WORD_ID_SEQ","CHAR_ID_SEQ","UPVOTES","ANSWERS","TOPQUALANS"]
    dfdata_question_info = dataset_to_dataframe(datax,datay,arrayoffeaturenames,0)

    filenameinfotrain = "data/user_info.txt"
    datax,datay = parsefile(filenameinfotrain,0)
    arrayoffeaturenames =["EID","ETAG","USER_WORD_ID_SEQ","USER_CHAR_ID_SEQ"]
    dfdata_user_info = dataset_to_dataframe(datax,datay,arrayoffeaturenames,0)

    data = build_data(dfdata_qid_eid_class,dfdata_question_info,dfdata_user_info)
    data.to_csv("new_train.txt", sep=',', encoding='utf-8',index=False)
    #data.reindex(columns=['LABEL','QID', 'EID', 'QTAG', 'UPVOTES', 'ANSWERS','TOPQUALANS', 'ETAG'])

    print data.shape
    print data.columns

    print open("new_train.txt").read().strip().split("\n")[1:4]

    #===========================

    filenameinfovalidate = "data/validate_nolabel.txt"
    datax, datay = parsefile_val_and_test(filenameinfovalidate, 0)
    arrayoffeaturenames = ["QID", "EID"]
    dfdata_qid_eid_class_val = dataset_to_dataframe(datax[1:], datay, arrayoffeaturenames, 0)

    data = build_data(dfdata_qid_eid_class_val, dfdata_question_info, dfdata_user_info)
    data.to_csv("new_validate.txt", sep=',', encoding='utf-8', index=False)

    print data.shape
    print data.columns
    # ===========================

    filenameinfotest = "data/test_nolabel.txt"
    datax, datay = parsefile_val_and_test(filenameinfotest, 0)
    arrayoffeaturenames = ["QID", "EID"]
    dfdata_qid_eid_class_test = dataset_to_dataframe(datax[1:], datay, arrayoffeaturenames, 0)

    data = build_data(dfdata_qid_eid_class_test, dfdata_question_info, dfdata_user_info)
    data.to_csv("new_test.txt", sep=',', encoding='utf-8', index=False)

    print data.shape
    print data.columns
    # ===========================