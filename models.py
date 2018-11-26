from featurizer import NBFeaturer
import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline, make_pipeline
import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from scipy import sparse
import xgboost, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
from sklearn.pipeline import Pipeline, make_pipeline

from importlib import reload 
from sklearn.pipeline import FeatureUnion, make_union

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
# feature
comment_col = 'comment_text'
train_x = train['comment_text']
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000, ngram_range=(1,1))

final_featurizers = {}
for label_col in label_cols:
    print('featurizer: ', label_col)
    nbfeaturer = NBFeaturer(0.2)
    f = make_pipeline(tfidf_vect, nbfeaturer)
    train_y = train[label_col]
    f.fit(train_x, train_y)
    final_featurizers[label_col] = f

# train 
def train_model(label_col):
    train_y = train[label_col]
    nbtrain_x = final_featurizers[label_col].transform(train_x)

    m = linear_model.LogisticRegression()
    m.fit(nbtrain_x, train_y)
    return m

models = {}
for label_col in label_cols:
    print('train: ', label_col)
    models[label_col] = train_model(label_col)

# test
test_x = test[comment_col]
preds = np.zeros((len(test), len(label_cols)))
for i, label_col in enumerate(label_cols):
    print('test: ', label_col)
    final_featurizer = final_featurizers[label_col]
    nbtest_x = final_featurizer.transform(test_x)
    
    model = models[label_col]
    preds[:,i] = model.predict_proba(nbtest_x)[:,1]

# out
submid = pd.DataFrame({'id': test["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns = label_cols)], axis=1)
submission.to_csv('submission.csv', index=False)



