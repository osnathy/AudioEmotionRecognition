from os import listdir
from os.path import isfile, join
import scipy.io.wavfile
import re
import pandas as pd
import pickle
import numpy as np


emotions = {'ang': np.int32(0), 'exc': np.int32(1), 'fru': np.int32(2), 'hap': np.int32(3), 'neu': np.int32(4), 'sad': np.int32(5)}

def create_utterance_information_file(sessions_path, to_save_utterance_information_file_name):
    utterance_information_df = pd.DataFrame()
    for k in range(5):
        df = build_utterance_information("%s%s" % (sessions_path, k + 1))
        utterance_information_df = utterance_information_df.append(df)

    with open(to_save_utterance_information_file_name, 'wb') as file:
        pickle.dump(utterance_information_df, file, protocol=pickle.HIGHEST_PROTOCOL)


def build_utterance_information(session_path):
    utterance_information_df = pd.DataFrame()
    utterance_information_dict = {}
    path_to_emo_evaluation = session_path + '/dialog/EmoEvaluation/'
    path_to_wav = session_path + '/sentences/wav/'
    count = 0
    for emotion_file in [f for f in listdir(path_to_emo_evaluation) if isfile(join(path_to_emo_evaluation, f))]:
        print(count)
        count += 1
        for utterance in (get_utter_info_from_eval_file(path_to_emo_evaluation + emotion_file)):
            if ((utterance[3] == 'neu')
                or (utterance[3] == 'hap')
                or (utterance[3] == 'sad')
                or (utterance[3] == 'ang')
                or (utterance[3] == 'exc')
                or (utterance[3] == 'fru')
            ):

                path = path_to_wav + utterance[2][:-5] + '/' + utterance[2] + '.wav'
                (rate, signal) = scipy.io.wavfile.read(path, mmap=False)
                utterance_information_dict = {"key": utterance[2], "signal": signal, "rate": rate, "label": emotions[utterance[3]]}
                utterance_information_df = utterance_information_df.append(utterance_information_dict, ignore_index=True)

    return utterance_information_df

def get_utter_info_from_eval_file(input_file):
    pattern = re.compile('[\[]*[0-9]*[.][0-9]*[ -]*[0-9]*[.][0-9]*[\]][\t][a-z0-9_]*[\t][a-z]{3}[\t][\[][0-9]*[.][0-9]*[, ]+[0-9]*[.][0-9]*[, ]+[0-9]*[.][0-9]*[\]]',re.IGNORECASE)
    with open(input_file, "r") as my_file:
        data = my_file.read().replace('\n', ' ')
    result = pattern.findall(data)
    utterance_list = []
    for i in result:
        a = i.replace('[', '')
        b = a.replace(' - ', '\t')
        c = b.replace(']', '')
        x = c.replace(', ', '\t')
        utterance_list.append(x.split('\t'))
    return utterance_list



