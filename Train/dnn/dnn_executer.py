import pickle

import keras
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.callbacks import History
from keras.callbacks import EarlyStopping

from visualization.data_visualization import show_loss, show_accuracy


def run_dnn(segment_level_features_file_name_to_load):
    training_epoch = 10
    batch_size = 128

    with open(segment_level_features_file_name_to_load, 'rb') as handle:
        data = pickle.load(handle)

    x = data[:, :650]
    y = np.take(data, [650], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    print("==================================================================")
    history = History()

    model = get_model()
    one_hot_labels = keras.utils.to_categorical(y_train, num_classes=6)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

    records = model.fit(x_train, one_hot_labels,
                        batch_size=batch_size,
                        nb_epoch=training_epoch,
                        validation_split=0.30,
                        verbose=1,
                        callbacks=[history, es])

    print(records.history.keys())

    # save the model
    model_json = model.to_json()
    with open("dnn_model.json", "w") as json_file:
        json_file.write(model_json)

    # save the model weights
    model.save_weights("dnn_model_weights.h5")

    # visualisation
    show_loss(records)
    show_accuracy(records)

    # evaluate the  data
    test_one_hot_labels = keras.utils.to_categorical(y_test, num_classes=6)
    accuracy = model.evaluate(x_test, test_one_hot_labels, batch_size=batch_size)
    print(accuracy)



    #TODO: CALCULATE  RECALL , PRESESION AND F1 SCORE FOR THE MODEL



# todo: 2. its 650 instead of 750 because we didnt calculate the snr features
def get_model():
    model = Sequential()
    # 750 its #of the dimensionality of the data
    # Adding the input layers and the first hidden layer
    model.add(Dense(output_dim=256, init='uniform', activation='relu', input_dim=650))

    # Adding the second + third hidden layer
    model.add(Dense(output_dim=256, init='uniform', activation='relu'))
    model.add(Dense(output_dim=256, init='uniform', activation='relu'))

    # output layer
    model.add(Dense(output_dim=6, init='uniform', activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print(model.summary())

    return model


