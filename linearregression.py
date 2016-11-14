from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import csv
import pandas as pd

namestest = ["QID","EID","QTAG","Q_WORD_ID_SEQ","UPVOTES","ANSWERS","TOPQUALANS","ETAG","E_WORD_ID_SEQ"]
namestrain = ["QID","EID","LABEL","QTAG","Q_WORD_ID_SEQ","UPVOTES","ANSWERS","TOPQUALANS","ETAG","E_WORD_ID_SEQ"]#"Q_CHAR_ID_SEQ","E_CHAR_ID_SEQ"]
namestestnow = [ "QTAG","ETAG","UPVOTES","ANSWERS","TOPQUALANS","Q_WORD_ID_SEQ","E_WORD_ID_SEQ"]

def seq ( row,numberofetags):
    newrow =[0]*numberofetags

    for x in row:
        #print x
        if x!='':
            newrow[int(x)]=1
    return newrow

def func1(row,numberofetags):
    # newrow =[0]*numberofetags
    #
    # for x in row:
    #     #print x
    #     if x!='':
    #         newrow[int(x)]=1
    return [int(x)if x!='' and x in row else 0 for x in range(0,numberofetags) ]



def removestuff(toremove,datax,dataxnew):
    dataxqwordid = datax[toremove].str.split("/",expand=True).stack().unique()
    data_new = [int(x) for x in dataxqwordid if x!='']
    numberofetags = max(data_new)+1
    print numberofetags
    blah =  datax[toremove].str.split("/").apply(func1,args=(numberofetags,) )
    print "aitha?"
    dataxnew2 = pd.concat([dataxnew,blah],axis=1)
    dataxnew3 = dataxnew2.drop([toremove],axis=1)
    return dataxnew3

def make01features(datax):
    dataxQTagnew = pd.get_dummies(datax["QTAG"],prefix="QTAG_")
    dataxnew = datax.drop(['QTAG'],axis=1)
    dataxnew = pd.concat([dataxnew,dataxQTagnew],axis=1)
    data1 = removestuff("ETAG",datax,dataxnew)
    data2 = removestuff("Q_WORD_ID_SEQ",datax,data1)
    #data3 = removestuff("E_WORD_ID_SEQ",datax,data1)
    return data2

def getoutputfiles(filename, outputfilename,regr):
    validate_x = pd.read_csv(filename, sep=',', names=namestest);
    validate_xdata = validate_x[namestestnow]
    validate_xdata = make01features(validate_xdata)
    label = pd.DataFrame(regr.predict_proba(validate_xdata)[:,:1],columns=["LABEL"])
    qid = validate_x["QID"]
    uid = validate_x["EID"]
    with open(outputfilename, 'wb') as mycsvfile:
        thedatawriter = csv.writer(mycsvfile,delimiter=",")
        thedatawriter.writerow(('qid','uid','label'))
        rows = zip(qid, uid, label["LABEL"])
        for row in rows:
            thedatawriter.writerow(row)

if __name__ == "__main__":
    filenametrain = "train.txt"
    data = pd.read_csv(filenametrain, sep=',', names=namestrain);
    datax = data[namestestnow]
    datay = data["LABEL"]
    print "vasihnvai 1"
    datax1 = make01features(datax)
    x_train,x_test,y_train,y_test= train_test_split(datax1, datay, test_size = 0.2)

    regr = LogisticRegression()
    regr.fit(x_train, y_train)
    print "vasihnvai 2"
    print "Training accuracy : " + str(regr.score(x_train, y_train))
    predicted = regr.predict(x_test)
    print "Testing accuracy : " + str(regr.score(x_test, y_test))

    #get the output files for submission
    filenamevalidation = "validate.txt"
    getoutputfiles(filenamevalidation,"temp.csv",regr)
    filenametest ="test.txt"
    getoutputfiles(filenamevalidation,"final.csv",regr)
