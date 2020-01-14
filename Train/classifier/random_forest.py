import numpy as np
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def run_svm(segment_level_features_file_name_to_load):
    with open(segment_level_features_file_name_to_load, 'rb') as handle:
        data = pickle.load(handle)

    x = data[:, :650]
    y = np.take(data, [650], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.4, random_state=0)
    clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0, n_jobs=-1)
    clf.fit(x_train, y_train.ravel())

    # Predict
    pred_probs = clf.predict_proba(x_test)

    print(pred_probs)
    # Results
    #display_results(y_test, pred_probs)

def evaluate(clf, x_train, x_test, y_train, y_test):
    y_pred = clf.predict(x_train)
    accuracy = accuracy_score(y_train, y_pred)
    classification_report_ = classification_report(y_train, y_pred)
    conf_matrix = confusion_matrix(y_train, y_pred)
    print("\nTraining: \nEpoch {} summary:\nAccuracy: {}\nClassification report:\n{}confusion_matrix:\n{}".format(accuracy,
                                                                                                                  classification_report_,
                                                                                                                  conf_matrix))
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    classification_report_ = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("\nTraining: \nEpoch {} summary:\nAccuracy: {}\nClassification report:\n{}confusion_matrix:\n{}".format(accuracy,
                                                                                                                  classification_report_,
                                                                                                                  conf_matrix))

