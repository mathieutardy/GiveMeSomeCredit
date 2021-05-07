# Useful packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

# Models
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Model Preprocessing and validation
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler

# Evaluation metric
from sklearn.metrics import roc_auc_score


class ELCreditScorer:
    def __init__(self):
        self.X, self.y = self.load_data("./data/cs-training.csv")
        self.classifier = None

    def load_data(self, path, train=True):
        df = pd.read_csv(path)
        # Replace missing values with mean of of column (applies to MonthlyIncome and MonthlyIncome).
        # It makes sense to replace by mean for those two variables
        df = df.fillna(df.mean())

        # Preprocessing
        self.X = df.drop(["SeriousDlqin2yrs", "Unnamed: 0"], axis=1).values
        scaler = StandardScaler().fit(self.X)
        self.X = scaler.transform(self.X)

        if train == True:
            self.y = df["SeriousDlqin2yrs"].values
            return self.X, self.y
        else:
            id = df["Unnamed: 0"].values
            return self.X, id

    def evaluate_classifiers(self):
        """ Evaluation of 9 classifiers """

        classifiers = [
            # SVC(),
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
            auc = cross_val_score(clf, self.X, self.y,
                                  cv=5, scoring="roc_auc").mean()
            if auc > best_auc:
                best_auc = auc
                best_clf = clf
            name = clf.__class__.__name__
            lst.append([name, auc])

        results = pd.DataFrame(lst, columns=["Classifier", "AUC"])
        results = results.sort_values(by="AUC", ascending=False)
        print(results)

        self.classifier = best_clf

        print("Evaluation is done.")

    def save_model(self, path):
        pickle.dump(self.classifier, open(path, "wb"))

    def train_classifier(self, path, classifier):
        """Train a specific classifier

        Args:
            path (string): path of where to save the model
        """
        self.classifier = classifier
        self.classifier.fit(self.X, self.y)
        auc = cross_val_score(
            self.classifier, self.X, self.y, cv=5, scoring="roc_auc"
        ).mean()
        print("AUC is: {0:.04f}".format(auc))
        self.save_model(path)

    def hyperparameter_tuning(self):
        """ Using Random Search CV we evaluable a large range of possible hyperparameters """

        # Set manually some parameters
        self.classifier.n_estimators = 300
        self.classifier.learning_rate = 0.01
        self.classifier.objective = "binary"

        random_params = {
            "max_depth": [6, 7, 8, 9, 10],
            "min_child_samples": [5, 10, 15, 20, 25, 30, 35, 40],
            "num_leaves": [10, 20, 30, 40, 50, 60],
        }

        clf = self.classifier
        random_lgb = RandomizedSearchCV(
            estimator=clf,
            param_distributions=random_params,
            scoring="roc_auc",
            cv=5,
            n_jobs=-1,
        )
        random_lgb.fit(self.X, self.y)

        print("Hyperparameter tuning done.")
        print("Best parameters are: {}".format(random_lgb.best_params_))
        print("Best score is: {0:.4f}".format(random_lgb.best_score_))

        self.classifier = random_lgb.best_estimator_

    def funetuning_model(self):
        """ Using GridSearchCV we finetune our hyperparameter space after after having applied a random search """

        grid_params = {
            "min_child_samples": [30, 35, 40, 45],
            "num_leaves": [50, 55, 60, 65],
        }

        clf = self.classifier

        grid_lgb = GridSearchCV(
            estimator=clf, param_grid=grid_params, scoring="roc_auc", cv=5, n_jobs=-1
        )

        grid_lgb.fit(self.X, self.y)

        print("Finetuning done.")
        print("Best parameters are: {}".format(grid_lgb.best_params_))
        print("Best score is: {0:.4f}".format(grid_lgb.best_score_))

        self.classifier = grid_lgb.best_estimator_

        self.save_model("./models/best_model.pkl")

    def load_model(self, path):
        self.classifer = pickle.load(open(path, "rb"))

    def plot_feature_importance(self):
        df = pd.read_csv("./data/cs-training.csv")
        columns = df.drop(["SeriousDlqin2yrs", "Unnamed: 0"], axis=1).columns
        plt.barh(columns, self.classifier.feature_importances_)
        plt.show()

    def submit_kaggle(self, path):
        """Creates a submission file in the correct format for the kaggle competition

        Args:
            path (string): path and filename describing where the file will be saved in
        """
        X_test, id = self.load_data("./data/cs-test.csv", train=False)
        predictions = self.classifier.predict_proba(X_test)[:, 1]
        submission = pd.DataFrame(
            list(zip(id, predictions)), columns=["Id", "Probability"]
        )
        submission.to_csv(path, index=False)
