
from Train.segmentLevelFeaturesExtraction.segment_level_features_calculator import calculate_segment_level_features

if __name__ == '__main__':
    number_of_segments = 25

    utterance_data_file_name = "utterance_data_file_name.pickle"
    segment_level_features_file_name = "segment_level_features.pickle"
    calculate_segment_level_features(utterance_data_file_name, number_of_segments, segment_level_features_file_name)