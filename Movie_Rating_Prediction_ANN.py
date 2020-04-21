import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from keras import models
from keras.layers import Dense


sw = set(stopwords.words('english'))
ps = PorterStemmer()
le = LabelEncoder()
cv = CountVectorizer(max_df = 0.5, max_features = 50000)
tfidf = TfidfTransformer()

def load_data():
    dataset = pd.read_csv("movie_data/train.txt", sep = 'delimiter', header = None, engine = 'python')
    dataset.columns = ["review"]

    labels = []
    for i in range(25000):
        if i < 12500:
            labels.append("pos")
        else:
            labels.append("neg")

    dataset = dataset.assign(label=labels)
    dataset = shuffle(dataset)
    return dataset

def clean_text(sample):
    sample = sample.lower()
    sample = sample.replace("<br /><br />", "")
    sample = re.sub("[^a-zA-Z]+", " ", sample)
    sample = sample.split()
    sample = [ps.stem(s) for s in sample if s not in sw]
    sample = " ".join(sample)
    return sample

def testing():
    test = pd.read_csv("movie_data/test.txt", sep = 'delimiter', header = None, engine = 'python')
    test.columns = ["review"]
    test['cleaned_review'] = test['review'].apply(clean_text)
    X_test = test['cleaned_review']
    X_test = cv.transform(X_test)
    X_test = tfidf.transform(X_test)
    y_pred = model.predict(X_test)
    y_pred [y_pred >= 0.5] = 1
    y_pred = y_pred.astype('int')
    dict = {0: 'neg', 1: 'pos'}
    y_pred = [dict[p[0]] for p in y_pred]
    ids = np.arange(25000)
    final_matrix = np.stack((ids, y_pred), axis = 1)
    df = pd.DataFrame(final_matrix, columns = ['Id', 'label'])
    df.to_csv('y_pred.csv', index = False)

if __name__ == "__main__":
    dataset = load_data()
    y = dataset['label'].values
    y = le.fit_transform(y)
    dataset['cleaned_review'] = dataset['review'].apply(clean_text)
    corpus = dataset['cleaned_review'].values

    X = cv.fit_transform(corpus)
    X = tfidf.fit_transform(X)

    #  Neural Network
    model = models.Sequential()
    model.add( Dense(16, activation = "relu", input_shape = (X.shape[1],) ) )
    model.add( Dense(16, activation = "relu") )
    model.add( Dense(1, activation = "sigmoid") )

    model.compile(optimizer = 'rmsprop', loss = "binary_crossentropy", metrics = ['accuracy'])

    X_val = X[:2500]
    X_train = X[2500:]

    y_val= y[:2500]
    y_train = y[2500:]

    hist = model.fit(X_train, y_train, batch_size = 128, epochs = 4, validation_data = (X_val, y_val))

    result = hist.history

    plt.plot(result['val_accuracy'], label = "Val acc")
    plt.plot(result['accuracy'], label = "Train acc")
    plt.legend()
    plt.show()

    plt.plot(result['val_loss'], label = "Val loss")
    plt.plot(result['loss'], label = "Train loss")
    plt.legend()
    plt.show()

    testing()



