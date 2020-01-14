import os

from Train.dataPrePreparation.data_preparation import utterance_data_preparation

if __name__ == '__main__':

    original_data_set_directory = "data/IEMOCAP/Session"
    utterance_data_file_name = "utterance_data_file_name.pickle"
    root_path = os.getcwd()
    iemocap_sessions_path = os.path.join(root_path, original_data_set_directory)
    #utterance_data_preparation(iemocap_sessions_path, utterance_data_file_name)