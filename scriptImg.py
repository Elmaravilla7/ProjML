import os
import glob
import numpy as np
import cv2
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
import pickle as pkl
from tensorflow.keras.models import model_from_json
from sklearn.preprocessing import label_binarize



def loadD(PATH):
    data = []
    fnames = glob.glob(PATH + '//*_*.jpg')
    i, j = cv2.imread(fnames[0], 0).shape
    target = [os.path.basename(t).split("_")[0] for t in fnames]
    for f in fnames:
        img = cv2.imread(f, 0)
        img = cv2.resize(img, (i, j))
        data.append(img)
    return np.array(data).reshape(len(data), i * j) / 255, np.array(target).reshape(len(data)), i, j

def loadM(fname):
    return pkl.load(open(fname + '.model', 'rb'))

def loadNN(fname):
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model = model_from_json(open(fname + '.json').read())
    model.load_weights(fname + '.h5')
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def preNN(X, y, i, j):
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    y = to_categorical(y)
    y = np.argmax(y, axis=1)
    X = X.reshape((len(X), i, j, 1))
    return X,y

def trasformN(y_test,pred):
    diz = {'jeans': 0, 'shirts': 1, 'trousers': 2, 'watches': 3}
    y_test_n = []
    pred_n = []
    for i in y_test:
        y_test_n.append(diz[i])
    for j in pred:
        pred_n.append(diz[j])
    return label_binarize(y_test_n, classes=[0, 1, 2, 3]), label_binarize(pred_n, classes=[0, 1, 2, 3])


def accuratezza(clf,X_test,y_test,typeC):
    if (typeC == 1):
        pred = np.argmax(clf.predict(X_test), axis=-1)
    else:
        pred = clf.predict(X_test)
    return accuracy_score(y_test, pred)



if __name__ == '__main__':
    #PATH = "/home/emiliocasella/Scrivania/proj204898/immagini-2/"
    #PATH2 = "./"
    PATH=input("Inserisci percorso cartella Validation Set ")
    PATH2=input("Inserisci percorso cartella contenente i modelli ")
    X_test, y_test, i, j = loadD(PATH)
    print("Validazione modelli...")
    print("AdaBoost...")
    ada_model = loadM(PATH2+"AdaBoostClassifier")
    print(accuratezza(ada_model, X_test, y_test, 0))
    print("SVC...")
    svc_model = loadM(PATH2+"SVC")
    print(accuratezza(svc_model, X_test, y_test, 0))
    print("KNN...")
    knn_model = loadM(PATH2+"KNeighborsClassifier")
    print(accuratezza(knn_model, X_test, y_test, 0))
    print("NN...")
    nn_model = loadNN(PATH2+"NN")
    X_test, y_test=preNN(X_test, y_test,i,j)
    print(accuratezza(nn_model, X_test, y_test, 1))
    print("Fatto!")

