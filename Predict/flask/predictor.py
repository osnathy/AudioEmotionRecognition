import os
import pickle
import numpy as np
import pandas as pd
import scipy.io.wavfile
from Train.segmentLevelFeaturesExtraction.segment_level_features_calculator import calculate_audio_features


def calculate_speech_features(path):
    number_of_segments = 25
    (rate, signal) = scipy.io.wavfile.read(path, mmap=False)

    return calculate_audio_features(signal, rate, number_of_segments)


def predict(request):
    root_path = os.getcwd()

    request_body = pd.Series(request)


    utterance_features, matrix_size = calculate_speech_features(os.path.join(root_path, request_body['file_name']))
    print("The speech matrix shape is:  ", str(utterance_features.shape))
    # load the model from disk
    loaded_model = pickle.load(open(os.path.join(root_path, 'classification_model/random_forest_model.pickle'), 'rb'))

    # make prediction
    prediction_result = loaded_model.predict(utterance_features)
    print(prediction_result)

    print("prediction_result shape: ", str(prediction_result.shape))


    # calculate prediction score
    utterance_index = np.full((matrix_size, 1), 0)
    frame_label_list = np.full((matrix_size, 1), 2)
    data = np.concatenate((utterance_features, frame_label_list, utterance_index), axis=1)
    x = data[:, :650]
    y = np.take(data, [650], axis=1)
    score = loaded_model.score(x, y)


    print("The score: " + str(score))

    return pd.Series(prediction_result).to_json(orient='values')


