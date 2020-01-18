import os
import pickle

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

    # load the model from disk
    loaded_model = pickle.load(open(os.path.join(root_path, 'classification_model/random_forest_model.pickle'), 'rb'))

    # make prediction
    prediction_result = loaded_model.predict(utterance_features)
    print(prediction_result)

    return pd.Series(prediction_result).to_json(orient='values')


