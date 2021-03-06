import nltk
import pandas as pd
import numpy as np
from collections import Counter
from nltk.tokenize import TweetTokenizer
import random
import csv
from sklearn.metrics import fbeta_score


def tokenize(text):
    tweet_tokenize = TweetTokenizer()
    return tweet_tokenize.tokenize(text)

def get_corpus_vocabulary(traind_df_text):
    '''Write a function to return all the words in a corpus.
    '''
    counter = Counter()
    for text in traind_df_text:
        tokens = tokenize(text)
        counter.update(tokens)
    return counter


def text_to_bow(text, wrd2idx):
    feature = np.zeros(len(wrd2idx))
    tokens = tokenize(text)
    for token in tokens:
        if token in wrd2idx:
            pozitie = wrd2idx[token]
            feature[pozitie] += 1 / (len(feature))
    return feature


def corpus_to_bow(all_text, wd2idx):
    all_features = []
    for text in all_text:
        all_features.append(text_to_bow(text, wd2idx))
    all_features = np.array(all_features)
    return all_features


def write_prediction(out_file, predictions):
    with open(out_file, 'w') as fout:
        fout.write('id,label\n')
        start_id = 5001
        for idx, pred in enumerate(predictions):
            linie = str(start_id + idx) + ',' + str(pred) + '\n'
            fout.write(linie)


def split(data, percent):
    '''
    :param data: datele pentru care urmeaza sa aplicam split
    :param percent: procentul de date folosite pentru train ( restul raman pentru test) in numere intregi de la 1 la 100
    '''
    shuffeled_data = data.sample(frac=1).reset_index(drop=True)
    percented_train_data = int(percent * len(data) / 100)
    train_data = shuffeled_data[:percented_train_data]
    test_data = shuffeled_data[percented_train_data:]
    return train_data, test_data


def accuracy(predictions, labels):
    corecte = 0
    labels = labels.reset_index(drop=True)
    for index, prediction in enumerate(predictions):
        if prediction == labels[index]:
            corecte += 1
    return corecte / len(predictions)


def scorF(predictions, labels):
    fp = 0
    fn = 0
    tp = 0
    labels = labels.reset_index(drop=True)
    for index, prediction in enumerate(predictions):
        if prediction == labels[index] and prediction == 1:
            tp += 1
        if prediction == 0 and labels[index] == 1:
            fn += 1
        if prediction == 1 and labels[index] == 0:
            fp += 1
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return (2 * p * r) / (p + r)


def cross_validation(data, labels, k):
    segment_size = int(len(labels) / k)
    indici = np.arange(0, len(labels))
    random.shuffle(indici)
    for i in range(0, len(labels), segment_size):
        indici_valid = indici[i:i + segment_size]
        left_side = indici[:i]
        right_side = indici[i + segment_size:]
        indici_train = np.concatenate([left_side, right_side])
        train = data[indici_train]
        valid = data[indici_valid]
        y_train = labels[indici_train]
        y_valid = labels[indici_valid]
        yield train, valid, y_train, y_valid

def countElem(lst, x):
    count = 0
    for ele in lst:
        if ele == x:
            count = count + 1
    return count


def cele_mai_comune_cuvinte_din_reprezentarile_cu_cei_mai_multi_0(all_features, corpus,cuvinte_si_frecvente,how_many):
    all_new_features = []
    result_dict = {}
    for idx,feature in enumerate(all_features):
        if countElem(feature,0) > (250*len(feature)/251):
            all_new_features.append(corpus[idx])


    for feature in all_new_features:
        tokens = tokenize(feature)
        for word in tokens:
            result_dict.update({word:cuvinte_si_frecvente[word]})
    most_comm = pd.DataFrame.from_dict(Counter(result_dict).most_common(how_many))
    most_comm.to_csv('common_0_wr2i_1700_hwMany_1500_250_251.csv')

def confusion_matrix(real_labels,predict_labels):
    false_positive = 0
    false_negative = 0
    true_positive = 0
    true_negative = 0
    for idx,label in enumerate(real_labels):
        if label == 1 and predict_labels[idx] == 0:
            false_negative += 1
        elif label == 1 and predict_labels[idx] == 1:
            true_positive += 1
        elif label == 0 and predict_labels[idx] == 0:
            true_negative += 1
        elif label == 0 and predict_labels[idx] == 1:
            false_positive += 1
    confusion_matrix = np.zeros((2,2))
    confusion_matrix[0,0]  = true_negative;
    confusion_matrix[0,1] = false_positive;
    confusion_matrix[1,0] = false_negative;
    confusion_matrix[1,1] = true_positive;

    return confusion_matrix

def exist(most_comm, word):
    for tuple in most_comm:
        if tuple[0]  == word:
            return True

def get_representation(vocabulary, how_many):
    mydict = pd.read_csv('common_0_wr2i_3500_hwMany_1000_250_251.csv', skiprows = 1, header = None)
    mydict = dict(zip(list(mydict[mydict.columns.values.tolist()[1]]), list(mydict[mydict.columns.values.tolist()[2]])))
    for key in mydict.keys():
        vocabulary.update({key:mydict[key]})
    most_comm = vocabulary.most_common(how_many)
    wrd2idx = {}
    idx2wrd = {}
    for index, tuple in enumerate(most_comm):
        wrd2idx[tuple[0]] = index
        idx2wrd[index] = tuple[0]
    return wrd2idx, idx2wrd

#
# train_df = pd.read_csv('train.csv')
# test_df = pd.read_csv('test.csv')
# corpus = train_df['text']
# test_corpus = test_df['text']
#
#
# toate_cuvintele = get_corpus_vocabulary(corpus)
#
# wrd2index,index2wrd = get_representation(toate_cuvintele,700)
#
# all_features = corpus_to_bow(corpus,wrd2index)
# for feature in all_features:
#     test_features = corpus_to_bow(test_corpus,wrd2index)
#
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC
#
# clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
# clf.fit(all_features,train_df['label'].values)
#
# predictions = clf.predict(test_features)
# write_prediction('twelfth_submission.csv', predictions)




## realizeaza fisierul cu scorurile in urma cross validarii
train_df = pd.read_csv('train.csv')
corpus = train_df['text']
toate_cuvintele = get_corpus_vocabulary(corpus)

wrd2index, index2wrd = get_representation(toate_cuvintele, 700)

all_features = corpus_to_bow(corpus, wrd2index)

labels = train_df['label'].values

from sklearn.metrics import fbeta_score

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
conf_matrix = np.array((2,2))
with open("Rezultate_Cross_Validation.txt","a+") as file:
    string_de_scris = "Rezultate submisia 12 folosind common_0_wr2i_3500_hwMany_1000_250_251.csv cu 700 wrd2idx: \n ";
    scor_mean = []
    for train, valid, y_train, y_valid in cross_validation(all_features, labels,10):
        clf.fit(train, y_train)
        predictii = clf.predict(valid)
        scor = fbeta_score(y_valid, predictii, beta=1)
        string_de_scris += "\n\n" + str(scor) + "\n"
        scor_mean.append(scor)
        conf_matrix = np.add(conf_matrix,confusion_matrix(y_valid, predictii))
    string_de_scris += "\n scorul mediu = " + str(np.mean(scor_mean)) + " \n"
    string_de_scris += "   0      1" + "\n" + "0 "
    for idx, row in enumerate(conf_matrix):
        if idx == 0:
            string_de_scris += str(row) + "\n"
        if idx == 1:
            string_de_scris += "1 " + str(row) + "\n"
    file.write(string_de_scris)




