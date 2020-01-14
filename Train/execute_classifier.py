from Train.classifier.svm_executer import run_svm

if __name__ == '__main__':
    number_of_segments = 25

    segment_level_features_file_name = "segment_level_features.pickle"
    run_svm(segment_level_features_file_name)