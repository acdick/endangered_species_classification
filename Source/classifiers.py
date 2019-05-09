from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

def classify(classifiers, X_train, X_test, y_train, y_test):
    
    metrics = []
    
    for clf in classifiers:
        
        # fit and predict all classifiers
        clf['Classifier'].fit(X_train, y_train)
        clf['y_hat_train'] = clf['Classifier'].predict(X_train)
        clf['y_hat_test'] = clf['Classifier'].predict(X_test)
        
        # append all training metrics
        metrics.append({'Model':            clf['Model'],
                        'Split':            'Train',
                        'Accuracy':         accuracy_score(  y_train, clf['y_hat_train']),
                        'Precision':        precision_score( y_train, clf['y_hat_train'], average='weighted'),
                        'Recall':           recall_score(    y_train, clf['y_hat_train'], average='weighted'),
                        'F1 Score':         f1_score(        y_train, clf['y_hat_train'], average='weighted'),
                        'Confusion Matrix': confusion_matrix(y_train, clf['y_hat_train'])})
        
        # append all testing metrics
        metrics.append({'Model':            clf['Model'],
                        'Split':            'Test',
                        'Accuracy':         accuracy_score(  y_test, clf['y_hat_test']),
                        'Precision':        precision_score( y_test, clf['y_hat_test'], average='weighted'),
                        'Recall':           recall_score(    y_test, clf['y_hat_test'], average='weighted'),
                        'F1 Score':         f1_score(        y_test, clf['y_hat_test'], average='weighted'),
                        'Confusion Matrix': confusion_matrix(y_test, clf['y_hat_test'])})
        
    return classifiers, metrics

def grid_search(grid_searches, X_train, X_test, y_train, y_test, columns):
    
    metrics = []
    
    for gs in grid_searches:
        # fit and predict all grid_searches
        gs['Classifier'].fit(X_train, y_train)
        gs['y_hat_train'] = gs['Classifier'].predict(X_train)
        gs['y_hat_test']  = gs['Classifier'].predict(X_test)
        
        # append all training metrics
        metrics.append({'Model':            gs['Model'],
                        'Split':            'Train',
                        'Best Parameters':  gs['Classifier'].best_params_,
                        'Features':         list(zip(columns, gs['Classifier'].best_estimator_.coef_[0])),
                        'Accuracy':         accuracy_score(  y_train, gs['y_hat_train']),
                        'Precision':        precision_score( y_train, gs['y_hat_train'], average='weighted'),
                        'Recall':           recall_score(    y_train, gs['y_hat_train'], average='weighted'),
                        'F1 Score':         f1_score(        y_train, gs['y_hat_train'], average='weighted'),
                        'Confusion Matrix': confusion_matrix(y_train, gs['y_hat_train'])})
        
        # append all testing metrics
        metrics.append({'Model':            gs['Model'],
                        'Split':            'Test',
                        'Best Parameters':  gs['Classifier'].best_params_,
                        'Features':         list(zip(columns, gs['Classifier'].best_estimator_.coef_[0])),
                        'Accuracy':         accuracy_score(  y_test, gs['y_hat_test']),
                        'Precision':        precision_score( y_test, gs['y_hat_test'], average='weighted'),
                        'Recall':           recall_score(    y_test, gs['y_hat_test'], average='weighted'),
                        'F1 Score':         f1_score(        y_test, gs['y_hat_test'], average='weighted'),
                        'Confusion Matrix': confusion_matrix(y_test, gs['y_hat_test'])})
    
    return grid_searches, metrics