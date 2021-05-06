import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pickle

# Packages for model training
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import roc_auc_score

class CreditScorer:

    def __init__(self):
        self.X, self.y = self.load_data('cs-training.csv')
        self.classifier = None
        
    def load_data(self,path, train=True):
        df = pd.read_csv(path)
        # Replace missing values with mean of of column (applies to )
        df = df.fillna(df.mean())
        self.X = df.drop(['SeriousDlqin2yrs','Unnamed: 0'], axis=1).values
        scaler = StandardScaler().fit(self.X)
        self.X = scaler.transform(self.X)
        if train == True:
            self.y = df["SeriousDlqin2yrs"].values
            return self.X,self.y
        else:
            id = df["Unnamed: 0"].values
            return X,id


    def evaluate_classifiers(self):
        classifiers = [
            SVC(),
            LogisticRegression(),
            DecisionTreeClassifier(),
            RandomForestClassifier(),
            AdaBoostClassifier(),
            GradientBoostingClassifier(),
            XGBClassifier(),
            ExtraTreesClassifier(),
            LGBMClassifier(),
            ] 

        lst = []
        best_clf = None
        best_auc = 0

        for clf in tqdm(classifiers):
            clf.fit(self.X, self.y)
            auc = cross_val_score(clf, self.X, self.y, cv=5, scoring = 'roc_auc').mean()
            if auc > best_auc:
                best_auc = auc
                best_clf = clf
            name = clf.__class__.__name__
            lst.append([name, auc])

        results = pd.DataFrame(lst, columns=["Classifier", "AUC"])
        results = results.sort_values(by="AUC", ascending=False)
        print(results)

        self.classifier = best_clf

    def save_model(self,path):
        pickle.dump(self.classifier,open( path, "wb"))

    def train_classifier(self,path):
        clf = self.classifier
        X,y = self.load_data('cs-training.csv')
        clf.fit(X, y)
        auc = cross_val_score(clf, X, y, cv=5, scoring = 'roc_auc').mean()
        print('AUC is: {0:.04f}'.format(auc))
        save_model(clf,path)

    def hyperparameter_tuning(self):

        self.classifier.n_estimators = 300
        self.classifier.learning_rate = 0.01
        self.classifier.objective = 'binary'

        random_params = {
            'max_depth' : [6,7,8,9,10],
            'min_child_samples': [5,10, 15,20,25,30,35,40],
            'num_leaves': [10,20,30,40,50,60],
            }

        clf = self.classifier
        random_lgb =  RandomizedSearchCV(estimator=clf, param_distributions=random_params,scoring='roc_auc',cv=5,n_jobs=-1)
        random_lgb.fit(self.X,self.y)
        print('Best parameters are: {}'.format(random_lgb.best_params_))
        print('Best score is: {0:.4f}'.format(random_lgb.best_score_))

        self.classifier = random_lgb.best_estimator_

    def funetuning_model(self):
        grid_params = {
            'min_child_samples': [30,35,40,45],
            'num_leaves': [50,55,60,65],
            }

        clf = self.classifier

        grid_lgb =  GridSearchCV(estimator=clf, param_grid=grid_params,scoring='roc_auc',cv=5,n_jobs=-1)

        grid_lgb.fit(self.X,self.y)
        print('Best parameters are: {}'.format(grid_lgb.best_params_))
        print('Best score is: {0:.4f}'.format(grid_lgb.best_score_))

        self.classifier = grid_lgb.best_estimator_

        self.save_model("./models/best_model.pkl")
    
    def load_model(self,path):
        self.classifer = pickle.load(open(path, "rb" ) )

    def submit_kaggle(self,path):
        X_test,id = self.load_data('cs-test.csv', train=False)
        predictions = self.classifier.predict_proba(X_test)[:, 1]
        submission = pd.DataFrame(list(zip(id, predictions)), columns = ['Id', 'Probability'])
        submission.to_csv(path,index=False)