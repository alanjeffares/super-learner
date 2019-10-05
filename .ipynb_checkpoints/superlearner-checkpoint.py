# from sklearn.base import BaseEstimator, ClassifierMixin
import pandas as pd
# from IPython.display import display, HTML, Image
# import matplotlib.pyplot as plt
# from matplotlib import pyplot
# from random import randint
import numpy as np
# import sys
from sklearn import dummy
import os

from sklearn import svm
from sklearn import tree
from sklearn import metrics
from sklearn import ensemble
from sklearn import linear_model
from sklearn import neighbors
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

class SuperLearnerClassifier():
    
    """
    An ensemble classifier that uses heterogeneous models at the base layer and a aggregation model at the aggregation layer.


    Parameters
    ----------
    
    use_stacked_prob : bool, optional (default = False)
        Option to use probability estimates rather than classifiacations 
        for training at the stacked layer.
    
    stacked_classifier : string or None, optional (default = decision_tree)
        Choice of classifier on the stacked dataset Z. Options are: 
        "decision_tree", "logistic_regression", "k_nearest_neighbours", 
        "random_forest" or "most_frequent".
        
    estimators_to_remove : list, optional (default = [])
        Option to remove (in order to specify) one or more of the base
        estimators. Choose from: ["decision_tree", "random_forest", 
        "linear_svm", "bagging", "k_neighbours", "logistic_regression"]
        
    include_original_input : bool, optional (default = False)
        Include original input data, X, at the stacked layer.
         
         
    Attributes
    ----------

    estimators = A dictionary of the form {"model_name": model, ... } 
        storing all the trained based estimators.
        
    Z_classifier = sklearn model object of the stacked layer model.
        
    Z = Pandas DataFrame containing the stacked layer dataset Z (when: use_stacked_prob 
        = False)
        
    Z_prob = Pandas DataFrame containing the stacked layer dataset Z (when: 
        use_stacked_prob = True)
    

    See also
    --------
    
    ----------
    .. [1]  van der Laan, M., Polley, E. & Hubbard, A. (2007). 
            Super Learner. Statistical Applications in Genetics 
            and Molecular Biology, 6(1) 
            doi:10.2202/1544-6115.1309
         
         
    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import cross_val_score
    >>> clf = SuperLearnerClassifier()
    >>> iris = load_iris()
    >>> clf.fit(pd.DataFrame(iris.data), iris.target)
    >>> cross_val_score(clf, pd.DataFrame(iris.data), iris.target, cv=10)

    """
    # Constructor for the classifier object
    def __init__(self, use_stacked_prob = False, stacked_classifier = "decision_tree", estimators_to_remove = [],
                include_original_input = False):
        """Setup a SuperLearner classifier"""     
        self.decision_tree = tree.DecisionTreeClassifier(criterion="entropy", max_depth=7, min_samples_split=11)
        self.random_forest = ensemble.RandomForestClassifier(n_estimators=500, max_features = 4) #change_max_features
        self.bagging = ensemble.BaggingClassifier(base_estimator = tree.DecisionTreeClassifier(criterion="entropy"), n_estimators=10)     
        self.logistic_model = linear_model.LogisticRegression(multi_class='auto')
        self.k_nearest_neighbours = neighbors.KNeighborsClassifier(n_neighbors=5)
        self.linear_svc = svm.SVC(kernel="linear", C=1.0, probability=True)

        self.include_original_input = include_original_input
        self.use_stacked_prob = use_stacked_prob
        
        self.estimators = {"decision_tree":self.decision_tree, "random_forest":self.random_forest,
                           "bagging":self.bagging, "logistic_regression":self.logistic_model,
                           "k_nearest_neighbours":self.k_nearest_neighbours, "linear_svc":self.linear_svc}
        
        #can use any subset of the availabe estimators
        self.estimators = {key: value for key, value in self.estimators.items() if key not in estimators_to_remove}


        #stacked layer classifier
        if stacked_classifier == "decision_tree" or stacked_classifier == None:
            self.Z_classifier = tree.DecisionTreeClassifier(criterion="entropy")
        elif stacked_classifier == "logistic_regression":
            self.Z_classifier = linear_model.LogisticRegression()
        elif stacked_classifier == "k_nearest_neighbours":
            self.Z_classifier = neighbors.KNeighborsClassifier(n_neighbors=5)
        elif stacked_classifier == "random_forest":
            self.Z_classifier = ensemble.RandomForestClassifier(n_estimators= 500)
        elif stacked_classifier == "most_frequent":
            self.Z_classifier = dummy.DummyClassifier(strategy="most_frequent")
        else:
            raise ValueError('Error: Not known classifier for stacked layer classifier, check spelling')
        
        
    def fit(self, X, Y):
        """Build a SuperLearner classifier from the training set (X, y).
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples. 
        y : array-like, shape = [n_samples] 
            The target values (class labels) as integers or strings.
        Returns
        -------
        self : object
        """    
        
        results_list = []
        results_list_prob = []
        
        if self.use_stacked_prob == False:  # use classifications at stacked layer
            k_fold = KFold(5, shuffle=False, random_state=None)
            for k, (train, test) in enumerate(k_fold.split(X, Y)):  # looping through folds
                prediction_list = [] 
                for name, model in self.estimators.items():
                    model.fit(X.iloc[train,], Y[train])  # fitting each model to fold training data
                    pred = model.predict(X.iloc[test,])  # predicting each model on its folds test data
                    prediction_list.append(np.array(pred)) 
                fold_k = pd.DataFrame(prediction_list) 
                fold_k = fold_k.T
                results_list.append(fold_k) 
                
            self.Z = pd.concat(results_list).reset_index(drop = True)  
            self.Z.columns = self.estimators.keys()
            
            #include original inputs data?
            if self.include_original_input:
                X.reset_index(drop = True)
                self.Z = pd.concat([self.Z, X.reset_index(drop = True)], axis=1, join_axes=[self.Z.index])
            
            #fit Z_classifier to Z
            self.Z_classifier.fit(self.Z, Y)
           

        elif self.use_stacked_prob:
            k_fold = KFold(5, shuffle=False, random_state=None)
            for k, (train, test) in enumerate(k_fold.split(X, Y)):  # looping through folds
                prediction_df = pd.DataFrame() 
                for name, model in self.estimators.items(): 
                    model.fit(X.iloc[train,], Y[train])  # fitting each model to fold training data
                    pred_prob = model.predict_proba(X.iloc[test,])  # predicting each model on its folds test data
                    pred_prob = pd.DataFrame(pred_prob) 
                    prediction_df = pd.concat([prediction_df, pred_prob], axis=1) 
 
                fold_k_prob = prediction_df 
                results_list_prob.append(fold_k_prob) 

            self.Z_prob = pd.concat(results_list_prob).reset_index(drop = True)  
            
            #include original inputs data?
            if self.include_original_input:
                X.reset_index(drop = True)
                self.Z_prob = pd.concat([self.Z_prob, X.reset_index(drop = True)], axis=1, join_axes=[self.Z_prob.index])
            
            #fit decision tree to Z
            self.Z_classifier.fit(self.Z_prob, Y)

        #Now retrain all estimators using full dataset
        self.estimators = {key: model.fit(X,Y) for key, model in self.estimators.items()}

    def predict(self, X):
        """Predict class labels of the input samples X.
        Parameters
        ----------
        X : array-like matrix of shape = [n_samples, n_features]
            The input samples. 
        Returns
        -------
        p : array of shape = [n_samples, ].
            The predicted class labels of the input samples. 
        """
        
        results_list = []
        
        if self.use_stacked_prob == False:
            for name, model in self.estimators.items():
                results_list.append(np.array(model.predict(X)))

            stacked_layer = pd.DataFrame(results_list).T
            #check if predicting on input values too
            if self.include_original_input == False:
                pass
            else:
                X.reset_index(drop = True)
                stacked_layer = pd.concat([stacked_layer, X.reset_index(drop = True)], axis=1, join_axes=[stacked_layer.index])

        elif self.use_stacked_prob == True:
            for name, model in self.estimators.items():
                results_list.append(pd.DataFrame(model.predict_proba(X)))
        
            stacked_layer = pd.concat(results_list, axis=1)
            #check if predicting on input values too
            if self.include_original_input == False:
                pass
            else:
                X.reset_index(drop = True)
                stacked_layer = pd.concat([stacked_layer, X.reset_index(drop = True)], axis=1, join_axes=[stacked_layer.index])
        
        return self.Z_classifier.predict(stacked_layer)

