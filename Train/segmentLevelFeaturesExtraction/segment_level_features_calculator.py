
import numpy as np
import pickle
import librosa
from python_speech_features import mfcc
import Signal_Analysis.features.signal as sig


def calculate_segment_level_features(utterance_information_data_path, number_of_segments, to_save_segment_level_features_file_name):
    tmp_idx = 0
    dataset = np.zeros([1, 652])

    with open(utterance_information_data_path, 'rb') as uif:
        utterance_information = pickle.load(uif)
    uif.close()

    dataset = build_matrix_features(dataset, number_of_segments, tmp_idx, utterance_information)

    with open(to_save_segment_level_features_file_name, 'wb') as file:
        pickle.dump(dataset, file, protocol=pickle.HIGHEST_PROTOCOL)


def build_matrix_features(dataset, number_of_segments, tmp_idx, utterance_information):
    for index, row in utterance_information.iterrows():
        utterance_features, matrix_size = calculate_audio_features(row['signal'], row['rate'], number_of_segments)
        utterance_index = np.full((matrix_size, 1), tmp_idx)
        tmp_idx += 1

        frame_label_list = np.full((matrix_size, 1), row['label'])
        data = np.concatenate((utterance_features, frame_label_list, utterance_index), axis=1)
        dataset = np.concatenate((dataset, data), axis=0)
    return dataset


# TODO-NOT READY!!
def pitch_based_features(signal, rate, win_length, hop_length):
    mean_hnr = sig.get_HNR(signal, rate)
    return mean_hnr


def calculate_audio_features(signal, rate, segment_size):
    mfcc_features = mfcc(signal=signal, samplerate=rate, )
    delta1 = librosa.feature.delta(mfcc_features)
    extended_features = np.concatenate((mfcc_features, delta1), axis=1)

    #TODO : calc pitch period + HNR
    pitch_based = pitch_based_features(signal, rate, win_length=400, hop_length=160)

    segment_level_features = stack_features_segments(extended_features, segment_size)

    return segment_level_features, segment_level_features.shape[0]


def stack_features_segments(features, segment_size):
    one_side_segment_size = segment_size // 2
    result = features
    result = np.zeros(result.shape[1] * segment_size)
    for idx in range(features.shape[0]-segment_size+1):
        result = np.vstack([result, np.hstack(np.vstack([features[idx:idx + one_side_segment_size, :],
                                                         features[idx + one_side_segment_size, :],
                                                         features[idx + one_side_segment_size + 1:idx + 2 * one_side_segment_size + 1, :]
                                                         ]))])
    return result[1:, :]





