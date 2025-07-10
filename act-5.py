import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report,ConfusionMatrixDisplay, confusion_matrix 
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn import set_config
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import FitFailedWarning, ConvergenceWarning
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from pathlib import Path
from joblib import dump

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class ML: 
    def __init__(self,X,y, verbose = True, plot = True, regression = None, classification = None, undersample = None, oversample = None, dump = None):
        self.X = X
        self.y = y
        self.verbose = verbose
        self.plot = plot
        self.regression = regression
        self.classification = classification
        self.undersample = undersample
        self.oversample = oversample
        self.dump = dump
    
    def dumpfolder(self, file, type="model", filename=None):
        output_dir = Path("artifacts") / type 
        output_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            filename = f"{type}.pkl"
        try:
            dump(file , output_dir / filename)
            print(f"{type} se guardó correctamente en {output_dir/filename}")
        except Exception as e:
            print(f"Error al guardar {type}: {e}")

    def Preprocess(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        num_cols = X.select_dtypes(exclude=['object']).columns
        cat_cols = X.select_dtypes(include=['object']).columns

        preprocessor = ColumnTransformer(
            [
                ('num', StandardScaler(), num_cols),
                ('cat', OneHotEncoder(), cat_cols)
            ]
        )

        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)

        if self.dump:
             self.dumpfolder(
                  preprocessor, type = "preprocessor", filename = "preprocessor.pkl"
             )
        if self.classification:
            #Transformar las categorias del target
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)
            if self.dump:
                 self.dumpfolder(
                      le, type ="preprocessor", filename="labelencoder.pkl"
                 )
        elif self.regression:
             pass
        if self.oversample:
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train,y_train)
        if self.undersample:
            rus = RandomUnderSampler(random_state=42)
            X_train, y_train = rus.fit_resample(X_train,y_train)

        return X_train, X_test, y_train, y_test

    def LogReg(self,X_train, y_train, X_test, y_test):
            lr = LogisticRegression()
            lr_grid = [{
                    'penalty': ['l1', 'l2','none'],
                    'C' : [0.01, 0.1, 1, 10, 100],
                    'max_iter': [100, 200, 300]
                }
            ]

            grid_search = GridSearchCV(lr, lr_grid, cv=5, verbose=1)
            grid_search.fit(X_train, y_train)
            grid_best_params = grid_search.best_params_

            lrfinal = LogisticRegression(**grid_best_params)
            lrfinal.fit(X_train, y_train)

            if self.dump:
                 self.dumpfolder(
                      lrfinal, type="model", filename="logreg.pkl"
                 )
            y_pred = lrfinal.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            if self.verbose:
                print("\n")
                print("---------------------------------------------------")
                print("Regresion Logistica")
                print(f"Accuracy: {accuracy}")
                class_report = classification_report(y_test, y_pred)
                print(class_report)

            if self.plot:
                cm = confusion_matrix(y_test, y_pred)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot(cmap= 'Blues')
                plt.show()        

    def SVMMLI(self, X_train, y_train, X_test, y_test):
            svc = LinearSVC()
            svc_grid = [{
                    "C": [0.01, 0.1, 1, 10, 100],
                    "class_weight": [None, 'balanced'],
                    "max_iter": [100, 200, 300],
                    "loss": ['hinge', 'squared_hinge'],
                    "dual": [True, False]
                }
            ]

            grid_search = GridSearchCV(svc, svc_grid, cv=5, verbose=1)
            grid_search.fit(X_train, y_train)
            grid_best_params = grid_search.best_params_

            svcfinal = LinearSVC(**grid_best_params)
            svcfinal.fit(X_train, y_train)

            if self.dump:
                self.dumpfolder(
                    svcfinal, type="model", filename="svm_linear.pkl"
                )
            y_pred = svcfinal.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            if self.verbose:
                print("\n")
                print("---------------------------------------------------")
                print("SVM Lineal")
                print(f"Accuracy: {accuracy}")
                class_report = classification_report(y_test, y_pred)
                print(class_report)

            if self.plot:
                cm = confusion_matrix(y_test, y_pred)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot(cmap='Blues')
                plt.show()

    def SVMM(self, X_train, y_train, X_test, y_test):
        svc = SVC()
        svc_grid = [{
            "C": [0.01, 0.1, 1, 10, 100],
            "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
            "gamma": ['scale', 'auto'],
        }]

        grid_search = GridSearchCV(svc, svc_grid, cv=5, verbose=1)
        grid_search.fit(X_train, y_train)
        grid_best_params = grid_search.best_params_

        svcfinal = SVC(**grid_best_params)
        svcfinal.fit(X_train, y_train)

        if self.dump:
            self.dumpfolder(
                svcfinal, type="model", filename="svm.pkl"
            )
        y_pred = svcfinal.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        if self.verbose:
            print("\n")
            print("---------------------------------------------------")
            print("SVM")
            print(f"Accuracy: {accuracy}")
            class_report = classification_report(y_test, y_pred)
            print(class_report)

        if self.plot:
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap='Blues')
            plt.show()

    def DecisionTree(self, X_train, y_train, X_test, y_test):
        dtc = DecisionTreeClassifier()
        dtc_grid = [{
            "criterion": ["gini", "entropy"],
            "splitter": ["best", "random"],
            "max_depth": [None, 10, 20, 30],
            "class_weight": [None, 'balanced'],
        }]

        grid_search = GridSearchCV(dtc, dtc_grid, cv=5, verbose=1)
        grid_search.fit(X_train, y_train)
        grid_best_params = grid_search.best_params_

        dtcfinal = DecisionTreeClassifier(**grid_best_params)
        dtcfinal.fit(X_train, y_train)

        if self.dump:
            self.dumpfolder(
                dtcfinal, type="model", filename="decisiontree.pkl"
            )
        y_pred = dtcfinal.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        if self.verbose:
            print("\n")
            print("---------------------------------------------------")
            print("Árbol de Decisión")
            print(f"Accuracy: {accuracy}")
            class_report = classification_report(y_test, y_pred)
            print(class_report)

        if self.plot:
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap='Blues')
            plt.show()

    def RandomForest(self, X_train, y_train, X_test, y_test):
        rf = RandomForestClassifier()
        rf_grid = [{
            "criterion": ["gini", "entropy"],
            "class_weight": [None, 'balanced'],
            "warm_start": [True, False],
        }]

        grid_search = GridSearchCV(rf, rf_grid, cv=5, verbose=1)
        grid_search.fit(X_train, y_train)
        grid_best_params = grid_search.best_params_

        rffinal = RandomForestClassifier(**grid_best_params)
        rffinal.fit(X_train, y_train)

        if self.dump:
            self.dumpfolder(
                rffinal, type="model", filename="rff.pkl"
            )
        y_pred = rffinal.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        if self.verbose:
            print("\n")
            print("---------------------------------------------------")
            print("Random Forest")
            print(f"Accuracy: {accuracy}")
            class_report = classification_report(y_test, y_pred)
            print(class_report)

        if self.plot:
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap='Blues')
            plt.show()

    def XGBoostClassifierModel(self, X_train, y_train, X_test, y_test):
        xgb_clf = xgb.XGBClassifier()
        xgb_grid = [{
            "learning_rate": [0.01, 0.1, 0.2],
            "n_estimators": [50, 100, 200],
        }]
        grid_search = GridSearchCV(xgb_clf, xgb_grid, cv=5, verbose=1)
        grid_search.fit(X_train, y_train)
        grid_best_params = grid_search.best_params_

        xgbfinal = xgb.XGBClassifier(**grid_best_params)
        xgbfinal.fit(X_train, y_train)

        if self.dump:
            self.dumpfolder(
                xgbfinal, type="model", filename="xgb.pkl"
            )
        y_pred = xgbfinal.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        if self.verbose:
            print("\n")
            print("---------------------------------------------------")
            print("XGBoost Classifier")
            print(f"Accuracy: {accuracy}")
            class_report = classification_report(y_test, y_pred)
            print(class_report)

        if self.plot:
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap='Blues')
            plt.show()

    def AdaBoost(self, X_train, y_train, X_test, y_test):
        ad = AdaBoostClassifier()
        ad_grid = [{
                "learning_rate": [0.01, 0.1, 1, 10],    
                "n_estimators": [50, 100, 200],
                "algorithm": ['SAMME', 'SAMME.R'],
            }
        ]

        grid_search = GridSearchCV(ad, ad_grid, cv=5, verbose=1)
        grid_search.fit(X_train, y_train)
        grid_best_params = grid_search.best_params_

        adfinal = AdaBoostClassifier(**grid_best_params)
        adfinal.fit(X_train, y_train)

        if self.dump:
            self.dumpfolder(
                adfinal, type="model", filename="adaboost.pkl"
            )
        y_pred = adfinal.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        if self.verbose:
            print("\n")
            print("---------------------------------------------------")
            print("AdaBoost")
            print(f"Accuracy: {accuracy}")
            class_report = classification_report(y_test, y_pred)
            print(class_report)

        if self.plot:
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap='Blues')
            plt.show()

    def GradientBoost(self, X_train, y_train, X_test, y_test):
        gbc = GradientBoostingClassifier()
        gbc_grid = [{
            "learning_rate": [0.01, 0.1, 1, 10],
        }]

        grid_search = GridSearchCV(gbc, gbc_grid, cv=5, verbose=1)
        grid_search.fit(X_train, y_train)
        grid_best_params = grid_search.best_params_

        gbcfinal = GradientBoostingClassifier(**grid_best_params)
        gbcfinal.fit(X_train, y_train)

        if self.dump:
            self.dumpfolder(
                gbcfinal, type="model", filename="gradientboost.pkl"
            )
        y_pred = gbcfinal.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        if self.verbose:
            print("\n")
            print("---------------------------------------------------")
            print("Gradient Boost")
            print(f"Accuracy: {accuracy}")
            class_report = classification_report(y_test, y_pred)
            print(class_report)

        if self.plot:
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap='Blues')
            plt.show()

    def Run(self):
        X_train, X_test, y_train, y_test = self.Preprocess()

        self.LogReg(X_train, y_train, X_test, y_test)
        self.SVMMLI(X_train, y_train, X_test, y_test)
        self.SVMM(X_train, y_train, X_test, y_test)
        self.DecisionTree(X_train, y_train, X_test, y_test) 
        self.RandomForest(X_train, y_train, X_test, y_test)
        self.XGBoostClassifierModel(X_train, y_train, X_test, y_test)
        self.AdaBoost(X_train, y_train, X_test, y_test)
        self.GradientBoost(X_train, y_train, X_test, y_test)

test_pd = pd.read_csv("/home/lenovo/Escritorio/verano_patrones/actividades/test.csv")
train_pd = pd.read_csv("/home/lenovo/Escritorio/verano_patrones/actividades/train.csv")

df2 = pd.concat([train_pd, test_pd], axis=0)
df2.drop(['id'], axis=1, inplace=True)
df2.dropna(inplace=True)

X = df2.drop(['price_range'], axis=1)
y = df2['price_range']

MLTrain = ML(X,y,classification=True, dump = True, plot = False, verbose = True)
MLTrain.Run()