import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import io
import random
from datetime import datetime, date
import math

def dataset_to_dataframe(traindatax,traindatay ,  featurenames,yfeature):
    dfx = pd.DataFrame(traindatax,  columns=featurenames)
    if(yfeature==1):
        dfx ["CLASSES"] = traindatay
    return dfx

def parsefile(filename,yfeature):
    fp = open(filename,'r')
    datax =[]
    datay =[]
    for line in fp:
        data=line.split('\t')
        if(yfeature==1):
            datax.append(data[:-1])
            datay.append(data[-1])
        else:
            datax.append(data)
    return datax,datay

if __name__ == "__main__":
    filenameinfotrain = "invited_info_train.txt"
    datax,datay = parsefile(filenameinfotrain,1)
    arrayoffeaturenames =["QID","EID"]
    dfdata_qid_eid_class = dataset_to_dataframe(datax,datay,arrayoffeaturenames,1)

    filenameinfotrain = "question_info.txt"
    datax,datay = parsefile(filenameinfotrain,0)
    arrayoffeaturenames =["QID","QTAG","WORD_ID_SEQ","CHAR_ID_SEQ","UPVOTES","ANSWERS","TOPQUALANS"]
    dfdata_question_info = dataset_to_dataframe(datax,datay,arrayoffeaturenames,0)


    filenameinfotrain = "user_info.txt"
    datax,datay = parsefile(filenameinfotrain,0)
    arrayoffeaturenames =["EID","ETAG","WORD_ID_SEQ","CHAR_ID_SEQ"]
    dfdata_user_info = dataset_to_dataframe(datax,datay,arrayoffeaturenames,0)

    filenameinfotrain = "user_info.txt"
    datax,datay = parsefile(filenameinfotrain,0)
    arrayoffeaturenames =["QID","EID"]
    dfdata_validation_qid_eid = dataset_to_dataframe(datax,datay,arrayoffeaturenames,0)


