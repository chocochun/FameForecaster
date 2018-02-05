%matplotlib inline
import pandas as pd
import numpy as np
import pickle
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import sklearn.model_selection as xval
from sklearn.datasets.mldata import fetch_mldata
import forestci as fci
import sklearn
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn import metrics
import itertools

from collections import Counter

class Classfier():
    '''
    A class to combine data from different source.
    '''
    def __init__(self):        
        self.cleandata = pickle.load(open("/Users/minchunzhou/Desktop/insight/dataformodel.pickle", "rb"))
        
    def rf_classifier(self):
        
        seperate_year = self.year
        data_X = self.cleandata.drop(["yearOfBirth"] ,axis=1)
        
        # Walk forward
        data_X_train = data_X[  (data_X.workyear >= seperate_year-20)  &  (data_X.workyear <= seperate_year)   ]
        data_X_test = data_X[ (data_X.workyear > seperate_year) & (data_X.workyear <= seperate_year +20)  ]
        
        #data_X_train = data_X[  (data_X.workyear <= seperate_year)   ]
        #data_X_test = data_X[ (data_X.workyear > seperate_year)  ]

        data_y_test =  data_X_test.is_famous

        data_X_train = data_X_train.drop(["workyear"], axis=1)
        data_X_test = data_X_test.drop(["is_famous","workyear"], axis=1)

        # balance data
        famousdata = data_X_train[ data_X_train.is_famous == 1 ]
        infamousdata = data_X_train[ data_X_train.is_famous == 0 ]
        infamous_select = infamousdata.sample(famousdata.shape[0])
        data_X_train = famousdata.append(infamous_select)

        data_y_train = data_X_train.is_famous
        data_X_train = data_X_train.drop(["is_famous"], axis=1)
        self.names = data_X_train.columns

        # model
        rf = RandomForestClassifier(n_estimators=10)
        rf = rf.fit(data_X_train, data_y_train)
        predictions = rf.predict(data_X_test)

        pred_proba = rf.predict_proba(data_X_test)
        pred_proba = pd.DataFrame(pred_proba)
        fpr, tpr, thresholds = metrics.roc_curve(data_y_test, pred_proba.ix[:,1], pos_label=1)

        self.test_accuracy = accuracy_score(data_y_test, predictions)
        self.train_accuracy = accuracy_score(data_y_train, rf.predict(data_X_train))
        
        self.AUC = metrics.auc(fpr, tpr)
        cnf_matrix = confusion_matrix(data_y_test, predictions)
        self.confusion_matrix = cnf_matrix
        self.recall =  cnf_matrix[1,1].astype('float') / (cnf_matrix[1,0] + cnf_matrix[1,1]  )
        self.precision =  cnf_matrix[1,1].astype('float') / (cnf_matrix[0,1] + cnf_matrix[1,1]  )
        self.model = rf

        
    def model_overtime(self):
        
        self.allyear = range(1900,1961,5)
        self.recall_overtime = np.zeros(len(self.allyear))
        self.precision_overtime = np.zeros(len(self.allyear))
        self.AUC_overtime = np.zeros(len(self.allyear))
        self.test_accuracy_overtime = np.zeros(len(self.allyear))
        
        for i in range(0,len(self.allyear)):
            
            self.year = self.allyear[i]
            self.rf_classifier()
            
            self.recall_overtime[i] = self.recall
            self.precision_overtime[i] = self.precision
            self.AUC_overtime[i] = self.AUC
            self.test_accuracy_overtime[i] = self.test_accuracy
            
        
    def show_feature_importance(self):
        
        self.rf_classifier()
        print "Features sorted by their score:"
        print sorted(zip(map(lambda x: round(x, 4), self.model.feature_importances_), self.names), reverse=True)

    def plot_precision_recall(self):
        
        plt.plot(self.allyear,self.precision_overtime)
        plt.plot(self.allyear,self.recall_overtime)

        plt.ylabel('%')
        plt.xlabel('Year')

        plt.legend(['Precision', 'Recall'],
                   loc='lower left', fontsize=12)
   
    def plot_accuracy_AUC(self):
        
        plt.plot(self.allyear,self.test_accuracy_overtime)
        plt.plot(self.allyear,self.AUC_overtime)

        plt.ylabel('%')
        plt.xlabel('Year')

        plt.legend(['Test Accuracy', 'AUC'],
                   loc='lower left', fontsize=12)

if __name__ == '__main__':
    
    Classfier = Classfier()
    Classfier.model_overtime()    
        
