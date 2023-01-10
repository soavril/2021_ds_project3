import torch
import numpy as np
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"

name=f'eff_best_model.pt'
model=torch.load(name)
model.to(device)

model.eval()


validation_prediction_df = pd.DataFrame(columns=['labels'])
validation_prediction_df.index.name = 'id'
validation_true_labels_df = pd.DataFrame(columns=['labels'])
validation_true_labels_df.index.name = 'id'

with torch.no_grad():
    for idx in range(len(validation_set)):
        validation_sample = validation_set[idx]
        _, _, _, recording, labels = validation_sample
        out = model(torch.tensor(recording).unsqueeze(0).to(device)) # unsqueeze는 batch dimension을 추가해주기 위함
        sample_prediction = torch.nn.functional.sigmoid(out).squeeze() > 0.5 # Use 0.5 as a threshold / squeeze는 batch dimension을 제거해주기 위함
        indices_of_1s = np.where(sample_prediction.cpu())[0]
        str_indices_of_1s = ' '.join(map(str, indices_of_1s))
        validation_prediction_df.loc[idx] = [str_indices_of_1s]
        
        str_true_labels = ' '.join(labels)
        validation_true_labels_df.loc[idx] = [str_true_labels]
        
print(validation_prediction_df[:10])
