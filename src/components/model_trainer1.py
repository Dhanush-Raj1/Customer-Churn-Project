import os
import sys

from src.logger import logging
from src.exception_handling import CustomException
from src.utils import save_object, evaluate_models

from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


@dataclass
class ModelTrainerConfig:
    model_file_path = os.path.join('artifacts', "model.pkl")


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """
        Params: train, test array
        Returns: training the model, saving the best model and returns the accuracy score
        """
        try:
            logging.info("Model training has been started.")
            logging.info("Train, test split has been initiated")
            X_train, X_test, y_train, y_test = (train_array[:, :-1], test_array[:, :-1],
                                                train_array[:, -1], test_array[:, -1])

            models = { #"Logistic regression": LogisticRegression(),
                       #"Naive bayes": GaussianNB(),
                       #"Knn classifier": KNeighborsClassifier(),
                       #"Decision tree": DecisionTreeClassifier(),
                       #"Random forest": RandomForestClassifier(),
                       #"Adaboost classifier": AdaBoostClassifier(),
                       #"Xgboost classifier": XGBClassifier(),
                       "Catboost classifier": CatBoostClassifier(),
                       "Support vector classifier": SVC()  
                     }

            params = {
                #"Logistic regression": [ {'penalty': ['l2'], 'C': [0.01, 0.1, 1, 10, 100, 300],
                                          #'solver': ['liblinear', 'lbfgs', 'saga', 'newton-cholesky']},
                                         #{'penalty': ['l1'], 'C': [0.01, 0.1, 1, 10, 100, 300],
                                          #'solver': ['liblinear', 'saga']},
                                         #{'penalty': ['elasticnet'], 'C': [0.01, 0.1, 1, 10, 100, 300],
                                          #'solver': ['saga'], 'l1_ratio': [0.5]} ],
                
                #"Naive bayes": {'var_smoothing': [1e-09, 1e-08, 1e-07, 1e-06, 1e-07]},
                
                #"Knn classifier": {'n_neighbors': [3, 5, 7, 9, 11, 15, 20, 25, 30, 40, 50],
                                   #'weights': ['uniform', 'distance'],
                                   #'algorithm': ['brute', 'kd_tree', 'ball_tree']},
                
                #"Decision tree": {'criterion': ['gini', 'entropy', 'log_loss'],
                                  #'max_depth': [3, 5, 10, 15, 20, 25, 30, 40, 50],
                                  #'min_samples_split': [2, 5, 10],
                                  #'min_samples_leaf': [1, 2, 4],
                                  #'max_features': [None, 'sqrt', 'log2']},
                
                #"Random forest": {'n_estimators': [50, 100, 150, 200],
                                  #'criterion': ['gini', 'entropy', 'log_logg'],
                                  #'min_samples_split': [2, 5, 10],
                                  #'min_samples_leaf': [1, 2, 4],
                                  #'max_features': [None, 'sqrt', 'log2']},
                
                #"Adaboost classifier": {'estimator': [None, LogisticRegression(), KNeighborsClassifier()],
                                        #'n_estimators': [10, 50, 100, 200, 300, 400],
                                        #'learning_rate': [0.01, 0.1, 0.5, 1.0] },
                                               
                #"Xgboost classifier": {'n_estimators': [50, 100, 200],
                                       #'learning_rate': [0.01, 0.1, 0.2, 0.3, 0.6, 1.0],
                                       #'max_depth': [3, 5, 7, 10],
                                       #'sub_sample': [0.6, 0.8, 1.0],
                                       #'colsample_bytree': [0.6, 0.8, 1.0]},
                
                "Catboost classifier": {'iterations': [500],
                                        #'n_estimators': [100, 200, 300, 400],
                                        'depth': [4, 6, 8, 10],
                                        'learning_rate': [0.001, 0.05, 0.1],
                                        'l2_leaf_reg': [1, 3, 5, 7, 9],
                                        #'bagging_temperature': [0, 0.5, 1],
                                        'random_strength': [0, 0.5, 1, 1.5, 2]},
                                                                            
                
                "Support vector classifier": {'C': [0.1, 1, 10, 100],
                                              'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                                              'gamma': ['scale', 'auto']}   
                    
                }

            model_report = evaluate_models(X_train=X_train, y_train=y_train,
                                           X_test=X_test, y_test=y_test,
                                           models=models,
                                           params=params)
            
            best_model_name = max(model_report, key=lambda x: model_report[x]['test_accuracy'])
            best_model_score = model_report[best_model_name]['test_accuracy']
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info(f"Best found model is: {best_model_name}")

            save_object(file_path=self.model_trainer_config.model_file_path,
                        obj=best_model)

            predicted = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, predicted)

            print(f"Best Model: {best_model_name}")
            print(f"Final accuracy: {accuracy:.4f}")

            logging.info(f"Best Model: {best_model_name}")
            logging.info(f"Final accuracy: {accuracy:.4f}")
                         
            
            return accuracy

        except Exception as e:
            raise CustomException(e, sys)
