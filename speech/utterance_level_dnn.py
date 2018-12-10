import pickle

import keras
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split


def extract_utterance_level_features(segment_level_features_file_name_to_load):
    training_epoch = 50
    batch_size = 128

    with open(segment_level_features_file_name_to_load, 'rb') as handle:
        data = pickle.load(handle)

    x = data[:, :650]
    y = np.take(data, [650], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    print("==================================================================")

    classifier = Sequential()

    # 750 its #of the dimensionality of the data
    # Adding the input layers and the first hidden layer
    classifier.add(Dense(output_dim=256, init='uniform', activation='relu', input_dim=650))

    # Adding the second + third hidden layer
    classifier.add(Dense(output_dim=256, init='uniform', activation='relu'))
    classifier.add(Dense(output_dim=256, init='uniform', activation='relu'))
    # output
    classifier.add(Dense(output_dim=6, init='uniform', activation='softmax'))

    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    one_hot_labels = keras.utils.to_categorical(y_train, num_classes=6)
    classifier.fit(x_train, one_hot_labels, batch_size=batch_size, nb_epoch=training_epoch)

    test_one_hot_labels = keras.utils.to_categorical(y_test, num_classes=6)

    score = classifier.evaluate(x_test, test_one_hot_labels, batch_size=batch_size)
    print(score)
