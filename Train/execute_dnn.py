from Train.dnn.dnn_executer import train_dnn

if __name__ == '__main__':

    segment_level_features_file_name = "segment_level_features.pickle"
    train_dnn(segment_level_features_file_name)