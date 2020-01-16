from Train.classifier.random_forest import train_random_forest

if __name__ == '__main__':

    segment_level_features_file_name = "segment_level_features.pickle"
    train_random_forest(segment_level_features_file_name)