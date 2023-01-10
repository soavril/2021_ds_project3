import os
import numpy as np

def extract_age(info_file):
    '''
        info file(###.txt)로부터 나이 정보를 뽑아냅니다.
    '''
    with open(info_file, 'r') as f:
        info = f.read()
        for i, line in enumerate(info.split("\n")):
            if line.startswith("#Age"):
                age = float(line.split(": ")[1].strip())
    return age

def extract_sex(info_file):
    '''
        info file(###.txt)로부터 성별 정보를 뽑아냅니다.
    '''
    with open(info_file, 'r') as f:
            info = f.read()
            for i, line in enumerate(info.split("\n")):
                if line.startswith("#Sex"):
                    sex = line.split(": ")[1].strip()
    return sex


def extract_labels(info_file):
    '''
        info file(###.txt)로부터 label(들) 정보를 뽑아냅니다.
    '''
    with open(info_file, 'r') as f:
            info = f.read()
            for i, line in enumerate(info.split("\n")):
                if line.startswith("#Dx"):
                    labels = line.split(": ")[1].strip()
                    labels = labels.split()
    return labels

def read_files(data_directory, is_training=True):
    '''
        data directory(train 또는 test)로부터 모든 sample들의
        id, age, sex, recording, labels 정보를 읽어들여
        (id, age, sex, recording, labels)의 list를 반환합니다.
        is_training=False일 경우엔 labels 정보를 읽어들이지 않습니다.
    '''
    list_id = []
    list_age = []
    list_sex = []
    list_recording = []
    list_labels = []
    for f in os.listdir(data_directory):
        root, extension = os.path.splitext(f)
        if not root.startswith(".") and extension == ".txt":
            list_id.append(int(root))
            info_file = os.path.join(data_directory, root + ".txt")
            recording_file = os.path.join(data_directory, root + ".npy")
            age = extract_age(info_file)
            list_age.append(age)
            sex = extract_sex(info_file)
            list_sex.append(sex)
            with open(recording_file, 'rb') as g:
                recording = np.load(g)
                list_recording.append(recording)
            if is_training:
                labels = extract_labels(info_file)
                list_labels.append(labels)
    if is_training:
        return list(zip(list_id, list_age, list_sex, list_recording, list_labels))
    else:
        return list(zip(list_id, list_age, list_sex, list_recording))