from Train.dnn.dnn_executer import run_dnn

if __name__ == '__main__':
    number_of_segments = 25

    segment_level_features_file_name = "segment_level_features.pickle"
    run_dnn(segment_level_features_file_name)