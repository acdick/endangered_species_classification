from sklearn.model_selection import GridSearchCV

from sklearn.dummy        import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes  import MultinomialNB
from sklearn.neighbors    import KNeighborsClassifier
from sklearn.tree         import DecisionTreeClassifier
from sklearn.ensemble     import RandomForestClassifier
from sklearn.ensemble     import AdaBoostClassifier

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

import numpy             as np
import matplotlib.pyplot as plt
import seaborn           as sns
sns.set_style("darkgrid")

import pandas as pd

def grid_search_dummy_classifier(parameters):
    
    classifier = {'Classifier': 'Dummy',
                  'Grid Search': GridSearchCV(
                      DummyClassifier(),
                      parameters,
                      cv=5,
                      scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'],
                      refit='f1_weighted',
                      return_train_score=False,
                      verbose=1)}
    
    return classifier

def grid_search_logistic_regression(parameters):
    
    classifier = {'Classifier': 'Logistic Regression',
                  'Grid Search': GridSearchCV(
                      LogisticRegression(),
                      parameters,
                      cv=5,
                      scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'],
                      refit='f1_weighted',
                      return_train_score=False,
                      verbose=10,
                      n_jobs=-1)}
    
    return classifier

def grid_search_multinomial_nb(parameters):
    
    classifier = {'Classifier': 'Multinomial Naive Bayes',
                  'Grid Search': GridSearchCV(
                      MultinomialNB(),
                      parameters,
                      cv=5,
                      scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'],
                      refit='f1_weighted',
                      return_train_score=False,
                      verbose=10,
                      n_jobs=-1)}
    
    return classifier

def grid_search_k_neighbors_classifier(parameters):
    
    classifier = {'Classifier': 'K Nearest Neighbors',
                  'Grid Search': GridSearchCV(
                      KNeighborsClassifier(),
                      parameters,
                      cv=5,
                      scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'],
                      refit='f1_weighted',
                      return_train_score=False,
                      verbose=10,
                      n_jobs=-1)}
    
    return classifier

def grid_search_decision_tree_classifier(parameters):
    
    classifier = {'Classifier': 'Decision Tree',
                  'Grid Search': GridSearchCV(
                      DecisionTreeClassifier(),
                      parameters,
                      cv=5,
                      scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'],
                      refit='f1_weighted',
                      return_train_score=False,
                      verbose=10,
                      n_jobs=-1)}
    
    return classifier

def grid_search_random_forest_classifier(parameters):
    
    classifier = {'Classifier': 'Random Forest',
                  'Grid Search': GridSearchCV(
                      RandomForestClassifier(),
                      parameters,
                      cv=5,
                      scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'],
                      refit='f1_weighted',
                      return_train_score=False,
                      verbose=10,
                      n_jobs=-1)}
    
    return classifier

def grid_search_ada_boost_classifier(parameters):
    
    classifier = {'Classifier': 'Ada Boost',
                  'Grid Search': GridSearchCV(
                      AdaBoostClassifier(),
                      parameters,
                      cv=5,
                      scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'],
                      refit='f1_weighted',
                      return_train_score=False,
                      verbose=10,
                      n_jobs=-1)}
    
    return classifier

def fit_predict_measure(data_name, X_train, X_test, y_train, y_test, y_labels, classifiers):
    
    all_models = pd.DataFrame()
    
    for classifier in classifiers:
        
        # fit training set
        print('Running jobs: ' + classifier['Classifier'])
        classifier['Grid Search'].fit(X_train, y_train)
        
        # cross-validated training metrics
        train = pd.DataFrame(classifier['Grid Search'].cv_results_)
        
        train = train[['params',
                       'mean_test_accuracy',
                       'mean_test_precision_weighted',
                       'mean_test_recall_weighted',
                       'mean_test_f1_weighted']]
        
        train = train.rename(index=str, columns={'params':                       'Parameters',
                                                 'mean_test_accuracy':           'Accuracy',
                                                 'mean_test_precision_weighted': 'Precision',
                                                 'mean_test_recall_weighted':    'Recall',
                                                 'mean_test_f1_weighted':        'F1 Score'})
        
        train['Data']             = data_name
        train['Classifier']       = classifier['Classifier']
        train['Split']            = 'Train'
        all_models = all_models.append(train, ignore_index=True)
        
        # hold-out test performance for best estimators
        y_hat_test = classifier['Grid Search'].predict(X_test)
        
        all_models = all_models.append(
            {'Parameters':       classifier['Grid Search'].best_params_,
             'Accuracy':         accuracy_score( y_test, y_hat_test),
             'Precision':        precision_score(y_test, y_hat_test, average='weighted'),
             'Recall':           recall_score(   y_test, y_hat_test, average='weighted'),
             'F1 Score':         f1_score(       y_test, y_hat_test, average='weighted'),
             'Data':             data_name,
             'Classifier':       classifier['Classifier'],
             'Split':            'Test',
             'Confusion Matrix': confusion_matrix(y_test, y_hat_test, labels=y_labels)}, ignore_index=True)
        
    all_models = all_models[['Data',     'Classifier', 'Parameters', 'Split',
                             'Accuracy', 'Precision',  'Recall',     'F1 Score',
                             'Confusion Matrix']]
        
    return all_models

def plot_confusion_matrices(confusion_matrices, y_labels):
    fig, axes = plt.subplots(confusion_matrices.shape[0], confusion_matrices.shape[1], figsize=(15,32))
    
    for i in range(confusion_matrices.shape[0]):
        for j in range(confusion_matrices.shape[1]):
            cm = confusion_matrices.iloc[i][j]
            cm = cm.astype('float') / np.sum(cm)
            
            sns.heatmap(
                cm,
                cmap='Blues',
                cbar=False,
                annot=True,
                fmt='.0%',
                linewidths=.5,
                xticklabels=y_labels,
                yticklabels=y_labels,
                square=True,
                ax=axes[i,j])
            
    return fig, axes