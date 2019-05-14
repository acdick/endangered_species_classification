from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

def fit_predict_metrics(classifiers, X_train, X_test, y_train, y_test, columns):
    
    metrics = []
    
    for classifier in classifiers:
        
        # fit all classifiers
        classifier['Classifier'].fit(X_train, y_train)
        
        # predict and append all training metrics
        classifier['y_hat_train'] = classifier['Classifier'].predict(X_train)
        
        metrics.append({'Model':            classifier['Model'],
                        'Split':            'Train',
                        'Best Parameters':  classifier['Classifier'].best_params_,
                        'Best Estimator':   classifier['Classifier'].best_estimator_,
                        'Accuracy':         accuracy_score(  y_train, classifier['y_hat_train']),
                        'Precision':        precision_score( y_train, classifier['y_hat_train'], average='weighted'),
                        'Recall':           recall_score(    y_train, classifier['y_hat_train'], average='weighted'),
                        'F1 Score':         f1_score(        y_train, classifier['y_hat_train'], average='weighted'),
                        'Confusion Matrix': confusion_matrix(y_train, classifier['y_hat_train'])})
        
        # predict and append all testing metrics
        classifier['y_hat_test'] = classifier['Classifier'].predict(X_test)
        
        metrics.append({'Model':            classifier['Model'],
                        'Split':            'Test',
                        'Best Parameters':  classifier['Classifier'].best_params_,
                        'Best Estimator':   classifier['Classifier'].best_estimator_,
                        'Accuracy':         accuracy_score(  y_test, classifier['y_hat_test']),
                        'Precision':        precision_score( y_test, classifier['y_hat_test'], average='weighted'),
                        'Recall':           recall_score(    y_test, classifier['y_hat_test'], average='weighted'),
                        'F1 Score':         f1_score(        y_test, classifier['y_hat_test'], average='weighted'),
                        'Confusion Matrix': confusion_matrix(y_test, classifier['y_hat_test'])})
    
    return classifiers, metrics