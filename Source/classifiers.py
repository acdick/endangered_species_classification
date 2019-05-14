from sklearn.model_selection import GridSearchCV

from sklearn.dummy        import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes  import GaussianNB
from sklearn.neighbors    import KNeighborsClassifier
from sklearn.tree         import DecisionTreeClassifier
from sklearn.ensemble     import RandomForestClassifier
from sklearn.ensemble     import AdaBoostClassifier

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

import pandas as pd

def grid_search_dummy_classifier(parameters, scoring):
    
    classifier = {'Classifier': 'Dummy',
                  'Grid Search': GridSearchCV(
                      DummyClassifier(),
                      parameters,
                      cv=5,
                      scoring=scoring,
                      refit='f1_weighted',
                      return_train_score=False,
                      verbose=1)}
    
    return classifier

def fit_predict_measure(data_name, X_train, X_test, y_train, y_test, y_labels, classifiers):
    
    all_models = pd.DataFrame()
    
    for classifier in classifiers:
        
        # fit training set
        classifier['Grid Search'].fit(X_train, y_train)
        
        # cross-validated training metrics
        train = pd.DataFrame(classifier['Grid Search'].cv_results_)
        train = train[['params', 'mean_test_accuracy', 'mean_test_f1_weighted']]
        train['Data']       = data_name
        train['Classifier'] = classifier['Classifier']
        train['Split']      = 'Train'
        
        # hold-out test performance
        y_hat_test = classifier['Grid Search'].predict(X_test)
        accuracy_score(y_test, y_hat_test)
        test['Confusion Matrix'] = confusion_matrix(y_test, y_hat_test)
        
    return train