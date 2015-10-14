import sqlite3
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import f1_score
from datetime import datetime as dt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class subredditClassifier:
    '''
    Subreddit Classifier.
    '''

    def __init__(self, fes, db_path='sample.db'):

        #db path
        self.db_path = db_path

        #Open connection to database.
        self.con = sqlite3.connect(self.db_path)
        
        #Container for feature extractors.
        self.fes = fes

        #Singular value value decomposition for dimensionality reduction.
        self.svd = TruncatedSVD(n_components=50)

        self.ch2 = SelectKBest(chi2, k=50)

        #Initialize the standard scaler 
        self.scl = StandardScaler()

        #Classifier.
        self.enet = SGDClassifier(penalty='elasticnet',
                             alpha=0.01,
                             l1_ratio=0.5)

        #There are 7 subreddit classes in our data set.
        #Which we'll encode as follows:
        self.classes = {
            u'funny': 0,
            u'nfl': 1,
            u'pics': 2,
            u'AskReddit': 3,
            u'leagueoflegends': 4,
            u'pcmasterrace': 5,
            u'nba': 6,
            }

        self.datasets = {
            'train': 0,
            'test': 1,
            'val': 2
        }

    def _result_iter(self, cursor, arraysize=1000, f=lambda x: x):
        while True:
            results = cursor.fetchmany(arraysize)
            if not results:
                break
            for result in results:
                yield f(result)

    def _get_X(self, dataset='train', limit=None, arraysize=1000):
        ds_id = self.datasets[dataset]
        cur = self.con.cursor()
        sql = 'SELECT body FROM May2015 WHERE length(body)>200 and dataset=' + str(ds_id)
        if limit is not None:
            sql += ' limit (' + str(limit) + ')'
        cur.execute(sql)
        return self._result_iter(cur, arraysize=arraysize, f=lambda x: x[0])
        
    def _get_y(self, dataset='train', limit=None):
        ds_id = self.datasets[dataset]
        cur = self.con.cursor()
        sql = 'SELECT subreddit FROM May2015 WHERE length(body)>200 and dataset=' + str(ds_id)
        if limit is not None:
            sql += ' limit (' + str(limit) + ')'
        cur.execute(sql)
        y = cur.fetchall()
        y = np.array(map(lambda x: self.classes[x[0]], y))
        return y

    def _X_y_iter(self, X, y, length = 1000):
        '''
        Params:
            X: sparse matrix
            y: ndarray
        '''
        n = y.shape[0]
        k = int(n/length)
        for i in xrange(0, k):
            X_sub = X[i*length : (i+1)*length, :]
            y_sub = y[i*length : (i+1)*length]
            yield (X_sub, y_sub)
        if n%length != 0:
            X_sub = X[k*length : , :]
            y_sub = y[k*length : ]
            yield (X_sub, y_sub)

    def fit(self):
        '''
        Train feature extractors and classifier.
        '''
        print(dt.now())

        #Iterators for training feature extractors.
        X_train_iter = self._get_X('train')
        
        #Train feature extractors
        print "Training feature extractor..." , dt.now()
        X_train = self.fes.fit_transform(X_train_iter)
        print "trained feature extractor : " , dt.now()

        print "Training feature selector..." , dt.now()
        X_train = self.svd.fit_transform(X_train)
        print "trained feature selector : " , dt.now()

        print "Training feature scaler..." , dt.now()
        X_train = self.scl.fit_transform(X_train)
        print "trained feature scaler : " , dt.now()

        print "getting y_train..." , dt.now()
        y_train = self._get_y('train')
        
        #Train classifier by stochastic gradient descent.
        print "Training classifier..." , dt.now()
        classes = self.classes.values()
        for (X, y) in self._X_y_iter(X_train, y_train):
            self.enet.partial_fit(X, y, classes=classes)

        print(dt.now())


    def test(self):
        print(dt.now())
        print("Testing...")

        X_test_iter = self._get_X('test')
        y_test = self._get_y('test')

        X_test = self.fes.transform(X_test_iter)
        X_test = self.svd.transform(X_test)
        X_test = self.scl.transform(X_test)

        score = self.enet.score(X_test, y_test)
        print "Accuracy =", score

        predictions = self.enet.predict(X_test)

        f1 = f1_score(y_test, predictions, average = None)
        print "F1_classes = ", f1
        f1 = f1_score(y_test, predictions, average = 'micro')
        print "F1_micro = ", f1
        f1 = f1_score(y_test, predictions, average = 'weighted')
        print "F1_weighted = ", f1

        print(dt.now())

if __name__ == '__main__':

    #Instantiate feature extractor.
    fe1 = CountVectorizer(min_df=3,  max_features=None, 
        strip_accents='unicode', analyzer='char_wb',
        ngram_range=(3, 3), stop_words = 'english')

    fe2 = TfidfVectorizer(min_df=3,  max_features=None, 
        strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
        ngram_range=(1, 2), use_idf=True,smooth_idf=True,sublinear_tf=True,
        stop_words = 'english')

    #Instantiate classifier with the desired feature extractor.
    cls = subredditClassifier(fes=fe2, db_path='/Users/Olga/Downloads/data/sample_db2.sqlite')

    #Fit classifier and feature extractors to the dataset.
    cls.fit()

    #Test feature extractor and classifier (weighted F1-score).
    cls.test()

    


    


