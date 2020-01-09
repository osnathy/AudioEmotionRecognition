import os

from DNN.dnn_executer import run_dnn
from DataPreProcessing.data_preparation import utterance_data_preparation
from SegmentLevelFeaturesExtraction.segment_level_features_calculator import calculate_segment_level_features

if __name__ == '__main__':
    number_of_segments = 25

    original_data_set_directory = "IEMOCAP/Session"
    utterance_data_file_name = "utterance_data.pickle"
    segment_level_features_file_name = "segment_level_features.pickle"
    root_path = os.getcwd()
    iemocap_sessions_path = os.path.join(root_path, original_data_set_directory)

    utterance_data_preparation(iemocap_sessions_path, utterance_data_file_name)
    calculate_segment_level_features(utterance_data_file_name, number_of_segments, segment_level_features_file_name)
    run_dnn(segment_level_features_file_name)


