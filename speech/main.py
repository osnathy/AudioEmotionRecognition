import os

from speech.segment_level_creator import build_segment_level_features
from speech.utterance_information_creator import create_utterance_information_file
from speech.utterance_level_dnn import extract_utterance_level_features

if __name__ == '__main__':
    number_of_segments = 25

    original_data_set_directory = "IEMOCAP/Session"
    utterance_information_file_name = "utterance_information_file_name.pickle"
    segment_level_features_file_name = "segment_level_features.pickle"
    root_path = os.getcwd()
    iemocap_sessions_path = os.path.join(root_path, original_data_set_directory)

    create_utterance_information_file(iemocap_sessions_path, utterance_information_file_name)
    build_segment_level_features(utterance_information_file_name, number_of_segments, segment_level_features_file_name)
    extract_utterance_level_features(segment_level_features_file_name)


