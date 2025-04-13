from sklearn.feature_extraction.text import TfidVectorizer

def vectorize(text):
    vector = TfidVectorizer()
    X = vector.fit_transform(text)
    return X, vector

