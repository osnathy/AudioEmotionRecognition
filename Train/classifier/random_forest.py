import numpy as np
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def train_random_forest(segment_level_features_file_name_to_load):
    with open(segment_level_features_file_name_to_load, 'rb') as handle:
        data = pickle.load(handle)

    print("Start splitting the data . . . ")
    x = data[:, :650]
    y = np.take(data, [650], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    print("Start initiating classifier object....")
    rf_clf = RandomForestClassifier(n_estimators=500, min_samples_split=25, random_state=0, n_jobs=-1)

    print("Start fitting . . . ")
    rf_clf.fit(x_train, y_train.ravel())
    print("Done fitting . . . ")

    print("Start saving the model . . . ")
    pickle.dump(rf_clf, open("random_forest_model.pickle", 'wb'))


    # calculate statistices for the model
    print("Start calculating statistics for the model . . . ")

    y_pred = rf_clf.predict(x_test)
    print("Random Forest Classifier report \n", classification_report(y_test, y_pred))







