from collections import OrderedDict
import pandas as pd
import pickle as pkl
import numpy as np
from keras.engine import training
from sklearn.linear_model import LogisticRegression
from itertools import chain
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.cross_validation import train_test_split
from ndcg import ndcg_at_k
import graphlab
import sys
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.scorer import make_scorer


def pickle(object, file_name):
    with open(file_name, 'wb') as f:
        pkl.dump(object, f)


def pickle_load(file_name):
    with open(file_name, 'r') as f:
        return pkl.load(f)


def transform_features(data, num_features):
    words = data.str.split('/').dropna().tolist()
    vocabulary = list(set(chain(*words)))
    if '' in vocabulary: vocabulary.remove('')
    print len(vocabulary)
    count_vect = CountVectorizer(vocabulary=vocabulary)
    tfidf_transformer = TfidfTransformer()

    words_transformed = []

    for word in words:
        words_transformed.append(" ".join(word))

    print "done"

    bow_count = count_vect.fit_transform(words_transformed)
    bow_tfidf = tfidf_transformer.fit_transform(bow_count)

    clf = TruncatedSVD(num_features)

    return pd.DataFrame(clf.fit_transform(bow_tfidf))


def transform_data(data, num_of_features, is_validate_set=False, running_on_desktop=False):
    to_drop = ['WORD_ID_SEQ', 'QID', 'EID', 'LABEL', 'ETAG', 'CHAR_ID_SEQ', 'USER_CHAR_ID_SEQ', 'USER_WORD_ID_SEQ']
    if is_validate_set:
        to_drop = ['WORD_ID_SEQ', 'QID', 'EID', 'ETAG', 'CHAR_ID_SEQ', 'USER_CHAR_ID_SEQ', 'USER_WORD_ID_SEQ']

    e_tag_features = 142

    if running_on_desktop:
        e_tag_features = 105

    return pd.concat(
        [data.drop(
            to_drop,
            axis=1),
            transform_features(data['WORD_ID_SEQ'], num_of_features),
            transform_features(data['USER_WORD_ID_SEQ'], num_of_features),
            transform_features(data['ETAG'], e_tag_features),
            #transform_features(data['CHAR_ID_SEQ'], num_of_features),
            #transform_features(data['USER_CHAR_ID_SEQ'], num_of_features)
        ], axis=1)


def evaluate(df_true, df_pred, column_name='QID', label_name='LABEL', method=0):
    print
    keys = df_true.groupby(column_name).groups
    score = 0.0
    count = 0

    for key in keys:
        true = df_true[df_true[column_name] == key][label_name]
        pred = df_pred[df_pred[column_name] == key][label_name]
        index = pred.sort_values(ascending=False).index
        r = true.reindex(index).tolist()
        ndcg5 = ndcg_at_k(r, 5, method)
        ndcg10 = ndcg_at_k(r, 10, method)
        result = (ndcg5 + ndcg10) / 2.0
        score += result
        count += 1
    return score / float(count)


def transform(series, dimension):
    row = series.index
    sentenses = series.str.split('/').dropna()
    model = gensim.models.Word2Vec(sentenses, workers=4, min_count=1, size=dimension)
    out = []
    for sentence in sentenses:
        result = np.zeros(dimension)
        count = 0
        for word in sentence:
            result += model[word]
            count += 1
        result = result / float(count)
        out.append(result)
    df = pd.DataFrame(out, index=sentenses.index).reindex(row).fillna(0)
    return df


def standardize_data(data):
    impute = Imputer()
    data = pd.DataFrame(impute.fit_transform(data), index=data.index)
    scaler = StandardScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(data))

    return scaled_data


def get_train_test_split(data, label):
    return train_test_split(data, label,
                            random_state=456,
                            test_size=0.2, stratify=label)


def svm(number_of_features, running_on_desktop=False):
    print "SVM  with Features = %d" % (number_of_features)
    data = pd.read_csv('new_train.txt', sep=',')
    if running_on_desktop:
        data = data[0:1000]
    transformed_train_data = transform_data(data, number_of_features, running_on_desktop=running_on_desktop)
    standardized_data = standardize_data(transformed_train_data)

    training_data, test_data, train_label, test_label = get_train_test_split(standardized_data, data['LABEL'])

    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]}]

    svm_ml = SVC(C=1, probability=True)
    svm_ml.fit(training_data, train_label)

    pred_probabality = pd.Series(svm_ml.predict_proba(test_data)[:, 1], index=test_data.index)

    df_true = pd.concat([data[['QID', 'EID']].ix[test_label.index, :],
                         pd.DataFrame(test_label, columns=['LABEL'])], axis=1)
    df_pred = pd.concat([data[['QID', 'EID']].ix[test_label.index, :],
                         pd.DataFrame(pred_probabality, columns=['LABEL'])], axis=1)

    print "NDGC Score = %f" % (evaluate(df_true, df_pred))

def graph_lab_reco(number_of_features, running_on_desktop=False):
    print "Graph lab reco with Features = %d" % (number_of_features)
    data = pd.read_csv('new_train.txt', sep=',')
    if running_on_desktop:
        data = data[0:1000]

    transformed_train_data = transform_data(data, number_of_features, running_on_desktop=running_on_desktop)
    transformed_train_data.columns = range(0,len(transformed_train_data.columns))


    #standardized_data = standardize_data(transformed_train_data)
    #standardize_data = transformed_train_data

    transformed_train_data["EID"] = data["EID"]
    transformed_train_data["QID"] = data["QID"]

    training_data, test_data, train_label, test_label = get_train_test_split(transformed_train_data, data['LABEL'])

    print len(training_data["QID"].tolist())
    print len(training_data["EID"].tolist())
    print len(test_label.tolist())
    sf = graphlab.SFrame({'item_id':training_data["QID"].tolist(),'user_id':training_data["EID"].tolist(),'ratings':train_label.tolist()})

    user_info_dict = {}
    for each_feature_column in (training_data.drop(["EID","QID"],axis=1)).columns.values[number_of_features+4:]:
        print each_feature_column
        user_info_dict["u_feature"+str(each_feature_column)] = list(training_data[each_feature_column])
        print len(list(training_data[each_feature_column]))
    user_info_dict["user_id"] = training_data["EID"].tolist()

    item_info_dict = {}

    for each_feature_column in (training_data.drop(["EID","QID"],axis=1)).columns.values[:number_of_features+3]:
        item_info_dict["i_feature"+str(each_feature_column)] = training_data[each_feature_column]
    item_info_dict["item_id"] = training_data["QID"].tolist()

    user_info = graphlab.SFrame(user_info_dict)
    item_info = graphlab.SFrame(item_info_dict)

    model = graphlab.factorization_recommender.create(sf, target='ratings', user_data=user_info, item_data=item_info,
                                                   nmf=True,binary_target=True)

    print model.recommend(exclude_known=False)
    #====================================
    sf_test = graphlab.SFrame({'item_id': test_data["QID"].tolist(), 'user_id': test_data["EID"].tolist()})

    user_info_dict = {}

    for each_feature_column in (test_data.drop(["EID", "QID"], axis=1)).columns.values[number_of_features + 4:]:
        user_info_dict["u_feature" + str(each_feature_column)] = list(test_data[each_feature_column])
        print len(list(test_data[each_feature_column]))
    user_info_dict["user_id"] = test_data["EID"].tolist()

    item_info_dict = {}

    for each_feature_column in (test_data.drop(["EID", "QID"], axis=1)).columns.values[:number_of_features + 3]:
        item_info_dict["i_feature" + str(each_feature_column)] = test_data[each_feature_column]
    item_info_dict["item_id"] = test_data["QID"].tolist()

    user_info = graphlab.SFrame(user_info_dict)
    item_info = graphlab.SFrame(item_info_dict)


    predictions = model.predict(sf_test,new_user_data=user_info,new_item_data=item_info)

    print list(predictions)


    # lr = LogisticRegression(n_jobs=-1)
    # lr.fit(training_data, train_label)
    #
    # pred_probabality = pd.Series(lr.predict_proba(test_data)[:, 1], index=test_data.index)
    #
    df_true = pd.concat([data[['QID', 'EID']].ix[test_label.index, :],
                         pd.DataFrame(test_label, columns=['LABEL'])], axis=1)
    df_pred = pd.concat([data[['QID', 'EID']].ix[test_label.index, :],
                         pd.DataFrame(list(predictions), columns=['LABEL'])], axis=1)

    print "NDGC Score = %f" % (evaluate(df_true, df_pred))
    #
    # lr = LogisticRegression()
    # lr.fit(transformed_train_data, data['LABEL'])
    #
    # validate_data = pd.read_csv('new_validate.txt', sep=',')
    # transformed_validate_data = transform_data(validate_data, number_of_features, is_validate_set=True,
    #                                            running_on_desktop=running_on_desktop)
    #
    # pred_probabality = pd.Series(lr.predict_proba(transformed_validate_data)[:, 1],
    #                              index=transformed_validate_data.index)
    #
    # df_pred = pd.concat([validate_data[['QID', 'EID']].ix[transformed_validate_data.index, :],
    #                      pd.DataFrame(pred_probabality, columns=['LABEL'])], axis=1)
    #
    # print df_pred
    #
    #df_pred.to_csv("validate_rs.csv", index=None)

def logistic_regression(number_of_features, running_on_desktop=False):
    print "Logistic Regression with Features = %d" % (number_of_features)
    data = pd.read_csv('new_train.txt', sep=',')
    if running_on_desktop:
        data = data[0:1000]
    transformed_train_data = transform_data(data, number_of_features, running_on_desktop=running_on_desktop)
    standardized_data = standardize_data(transformed_train_data)

    training_data, test_data, train_label, test_label = get_train_test_split(standardized_data, data['LABEL'])

    lr = LogisticRegression(n_jobs=-1)
    lr.fit(training_data, train_label)

    pred_probabality = pd.Series(lr.predict_proba(test_data)[:, 1], index=test_data.index)

    df_true = pd.concat([data[['QID', 'EID']].ix[test_label.index, :],
                         pd.DataFrame(test_label, columns=['LABEL'])], axis=1)
    df_pred = pd.concat([data[['QID', 'EID']].ix[test_label.index, :],
                         pd.DataFrame(pred_probabality, columns=['LABEL'])], axis=1)

    print "NDGC Score = %f" % (evaluate(df_true, df_pred))

    lr = LogisticRegression()
    lr.fit(transformed_train_data, data['LABEL'])

    validate_data = pd.read_csv('new_validate.txt', sep=',')
    transformed_validate_data = transform_data(validate_data, number_of_features, is_validate_set=True,
                                               running_on_desktop=running_on_desktop)

    pred_probabality = pd.Series(lr.predict_proba(transformed_validate_data)[:, 1],
                                 index=transformed_validate_data.index)

    df_pred = pd.concat([validate_data[['QID', 'EID']].ix[transformed_validate_data.index, :],
                         pd.DataFrame(pred_probabality, columns=['LABEL'])], axis=1)

    print df_pred

    df_pred.to_csv("validate_logistic.csv", index=None)


def main():
    pd.set_option('display.height', 1000)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    graph_lab_reco(100, running_on_desktop=False)

    # args = sys.argv
    #
    # if args[1] == 'logistic':
    #     logistic_regression(int(args[2]), running_on_desktop=True)
    # elif args[1] == 'svm':
    #     svm(int(args[2]), running_on_desktop=True)


if __name__ == "__main__":
    main()
