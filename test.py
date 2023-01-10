import pandas as pd
import numpy as np
import torch
from torch.utils import EfficientNet, get_model_params

model = EfficientNet.from_name('efficientnet-b4')

override={'num_classes':12, 'image_size': 5000}
blocks_args, global_params =get_model_params('efficientnet-b4', override)


device = "cuda" if torch.cuda.is_available() else "cpu"

model=EfficientNet(blocks_args, global_params)

model = model.to(device)
model.to(device)

model.eval()

test_prediction_df = pd.DataFrame(columns=['labels'])
test_prediction_df.index.name = 'id'

with torch.no_grad():
    for idx in range(len(test_set)):
        test_sample = test_set[idx]
        _, _, _, recording = test_sample
        out = model(torch.tensor(recording).unsqueeze(0).to(device)) # unsqueeze는 batch dimension을 추가해주기 위함
        sample_prediction = torch.nn.functional.sigmoid(out).squeeze() > 0.5 # Use 0.5 as a threshold / squeeze는 batch dimension을 제거해주기 위함
        indices_of_1s = np.where(sample_prediction.cpu())[0]
        str_indices_of_1s = ' '.join(map(str, indices_of_1s))
        test_prediction_df.loc[idx] = [str_indices_of_1s]
        
test_prediction_df[:10]

test_prediction_df.to_csv('my_submission.csv')