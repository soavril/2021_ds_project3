import torch
import numpy as np

#training sample을 batch 단위로 처리할 수 있도록 torch.uitls.data.Dataset을 이용한 custom dataset을 만들어 줍니다.

class Dataset_ECG(torch.utils.data.Dataset):
    """
        Build ECG dataset
    """
    def __init__(self, dataset, num_classes=12):
        """
            dataset을 읽어들여 id, age, sex, recording, labels를 저장한 list를 만들어 줍니다.
        """
        self.sample_id = []
        self.sample_age = []
        self.sample_sex = []
        self.sample_recording = []
        self.sample_labels = []
        self.num_samples = len(dataset)
        
        for idx in range(self.num_samples):
            _id, _age, _sex, _recording, _labels = dataset[idx]
            # model에 input으로 들어가는 data는 torch.Tensor 타입으로 변환해 줍니다.
            age = torch.tensor(_age//10)
            sex = torch.tensor([0,1]) if _sex == "F" else torch.tensor([1,0])
            recording = torch.tensor(_recording)
            labels = torch.tensor(np.zeros(num_classes))
            for label in _labels:
                labels[int(label)] = 1

            self.sample_id.append(_id)
            self.sample_age.append(age)
            self.sample_sex.append(sex)
            self.sample_recording.append(recording)
            self.sample_labels.append(labels)

        print(f'Loaded {self.num_samples} samples...')

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "id": self.sample_id[idx],
            "age": self.sample_age[idx],
            "sex": self.sample_sex[idx],
            "recording": self.sample_recording[idx],
            "labels": self.sample_labels[idx]
        }