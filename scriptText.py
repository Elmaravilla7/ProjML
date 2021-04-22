import glob
import numpy as np
import pandas as pd
import multiprocessing
from sklearn.metrics import accuracy_score
from tensorflow import keras
import pickle as pkl
from tensorflow.keras.models import model_from_json
import re
import string
import nltk
from nltk.corpus import stopwords
import gensim
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models.doc2vec import Doc2Vec
cores = multiprocessing.cpu_count()


def readFile(file):
    f = open(file, encoding="latin-1")
    return f.read()


def loadD(PATH1, PATH2):
    target = []
    data = []
    fnames1 = glob.glob(PATH1 + '*.txt')
    fnames2 = glob.glob(PATH2 + '*.txt')
    for f1 in fnames1:
        data.append(readFile(f1))
        target.append(0)  # ham
    for f2 in fnames2:
        data.append(readFile(f2))
        target.append(1)  # spam

    return pd.DataFrame(list(zip(data, target)),
                        columns=['text', 'target']).sample(frac=1).reset_index(drop=True)

def prep_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('http?://\S+|www\.\S+', '', text)
    text = re.sub('href', '', text)
    text = re.sub('subject', '', text)
    text = re.sub('re', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    text = tokenizer.tokenize(text)
    stop_words = stopwords.words('english')
    text = remove_stopwords(text, stop_words)
    return text



def remove_stopwords(text, stop_words):
    words = [w for w in text if w not in stop_words]
    return words


class MyTokenizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def transform(self, X, y):
        for c in y:
            for i in range(0, len(X[c])):
                X.at[i, c] = prep_text(X[c][i])


def contaP(tab,numW):
    words_freq={}
    data=[]
    frauTot = df[df['target'] == tab]
    text=frauTot['text']
    for i in text:
        for word in i:
            data.append(word)
    for key in data:
        if key not in words_freq:
            words_freq[key]=1
        else:
            words_freq[key]+=1
    sort=sorted(words_freq.items(), key=lambda x: x[1], reverse=True)
    return sort[:numW]



def cleanV(elements):
    modificata = np.zeros(305)
    for e in elements:
        modificata = np.vstack((modificata, e))
    return modificata[1:]


def createIndex(numWords):
    indexWords = {}
    words = []
    topWHam = contaP(0, numWords)
    topWSpam = contaP(1, numWords)
    for i in range(len(topWHam)):
        words.append(topWHam[i][0])
    for i in range(len(topWSpam)):
        words.append(topWSpam[i][0])
    i = 0
    for word in words:
        if word not in indexWords:
            indexWords[word] = i
            i = i + 1
    return indexWords


def createEmbeddings(text, indexWords):
    vector = np.zeros(len(indexWords), dtype=int)
    for word in text:
        if word in indexWords:
            i = indexWords[word]
            if (vector[i] == 0):
                vector[i] = 1
            else:
                vector[i] = vector[i] + 1
    return vector


class MyEmbeddings(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def transform(self, X, y):
        index = createIndex(200)
        for c in y:
            for i in range(0, len(X[c])):
                X.at[i, c] = createEmbeddings(X[c][i], index)



def loadM(fname):
    return pkl.load(open(fname+'.model', 'rb'))

def loadNN(fname):
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    model = model_from_json(open(fname+'.json').read())
    model.load_weights(fname+'.h5')
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model







def accuratezza(clf,X_test,y_test,typeC):
    if (typeC == 1):
        pred=(clf.predict(X_test) > 0.5).astype("int32")
    else:
        pred = clf.predict(X_test)
    return accuracy_score(y_test, pred)


def cleanV2(elements):
    modificata = np.zeros(500)
    for e in elements:
        modificata = np.vstack((modificata, e))
    return modificata[1:]


def convert_mean(text, model):
    vectors = model.infer_vector(text)
    return vectors


class MyDoc2vec(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y, model):
        return self

    def transform(self, X, y, fname):
        model = gensim.models.Doc2Vec.load(fname)
        for c in y:
            for i in range(0, len(X[c])):
                X.at[i, c] = convert_mean(X[c][i], model)

if __name__ == '__main__':
    #PATH1 = "/home/emiliocasella/Scrivania/proj204898/testi-2/ham/"
    #PATH2 = "/home/emiliocasella/Scrivania/proj204898/testi-2/spam/"
    #PATH3= "./"
    PATH1=input("Inserisci percorso cartella Validation Set cartella ham ")
    PATH2=input("Inserisci percorso cartella Validation Set cartella spam ")
    PATH3=input("Inserisci percorso cartella contenente i modelli ")
    print("Preprocessing dei dati...")
    df=loadD(PATH1,PATH2)
    print("Passo 1...")
    MyTokenizer().transform(df, ['text'])
    clean_word=df.copy()
    print("Passo 2...")
    MyEmbeddings().transform(df, ['text'])
    print("Passo 3...")
    X_test = cleanV(df['text'].to_numpy())
    y_test = df.target.to_numpy()
    print("Validazione modelli...")
    print("AdaBoost...")
    ada_model = loadM(PATH3+"AdaBoostClassifier")
    print(accuratezza(ada_model, X_test, y_test, 0))
    print("SVC...")
    svc_model = loadM(PATH3+"SVC")
    print(accuratezza(svc_model, X_test, y_test, 0))
    print("KNN...")
    knn_model = loadM(PATH3+"KNeighborsClassifier")
    print(accuratezza(knn_model, X_test, y_test, 0))
    print("NN...")
    nn_model = loadNN(PATH3+"NN")
    print(accuratezza(nn_model, X_test, y_test, 1))
    print("Embeddings con Doc2Vec + NN...")
    print("Creo rappresentazione vettoriale...")
    df=clean_word
    MyDoc2vec().transform(df, ['text'], "d2vec.model")
    X_test_doc2vec = cleanV2(df['text'].to_numpy())
    y_test_doc2vec = df.target.to_numpy()
    nn2_model = loadNN(PATH3 + "NN2DV")
    print(accuratezza(nn2_model, X_test_doc2vec, y_test_doc2vec, 1))
    print("Fatto!")



