import os
import torch
from help_function import *
from pytorch_custom_dataset import *
from efficient_model import *
import time
import datetime

#print(os.getcwd())
os.chdir('data_project3')

# cuda gpu를 사용할 수 있을 경우 사용합니다.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Training에 사용될 hyperparameter를 정해줍니다.
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 0.001

training_dir = 'data_project3/train'
test_dir = 'data_project3/test'

#training_dataset = torch.load('augmented_train_2.pt')
training_data = sorted(read_files(training_dir), key=lambda sample: sample[0])
#validation_set = torch.load('val.pt')
num_training = len(training_data)

training_dataset = Dataset_ECG(training_data, num_classes=12)

# Training dataset을 batch 단위로 읽어들일 수 있도록 DataLoader를 만들어줍니다.
training_loader = torch.utils.data.DataLoader(training_dataset,pin_memory=True, batch_size=BATCH_SIZE)

#training

model=EfficientNet(blocks_args, global_params)

model = model.to(device)
model.to(device)
model.train()
pos=[811, 1062, 958, 3204, 574, 515, 625, 1203, 10258, 1114, 2451, 563]
p=[15370-x for x in pos]
p_w=[x/y for x, y in zip(p, pos)]
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor(p_w).to(device)) # for multi-label classification
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

from torch.optim.lr_scheduler import PolynomialLRDecay

decay_steps = (num_training//BATCH_SIZE)*EPOCHS
scheduler_poly_lr_decay = PolynomialLRDecay(optimizer, max_decay_steps=decay_steps, end_learning_rate=1e-6, power=0.9)

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
import csv

def cal_run_time(sec):
    times = str(datetime.timedelta(seconds=sec)).split(".")
    return times[0]


def train_model(model, criterion, optimizer, scheduler, num_epochs=20):
    train_start = time.time()

    #best_model_wts = copy.deepcopy(model.state_dict())
    best_f1_score = 0.0
    epoch_loss, epoch_score = [], []
    # Each epoch has a training and validation phase
    for epoch in range(1, num_epochs+1):
        print('***** Epoch {}/{} *****'.format(epoch, num_epochs))
        start = time.time()

        ## Train
        model.train()
        train_loss_list = []
        epoch_training_loss_sum = 0.0
        for i_batch, sample_batched in enumerate(training_loader):
            b_recording = sample_batched["recording"].to(device)       
            #b_sex=sample_batched["sex"].to(device)
            #b_age=sample_batched["age"].to(device)
            b_labels = sample_batched["labels"].to(device)          
            optimizer.zero_grad()
            b_out = model(b_recording.float())                      
            loss = criterion(b_out, b_labels)
            loss.backward()
            optimizer.step()
            epoch_training_loss_sum += loss.item() * b_labels.shape[0]
            scheduler.step()
        
        # batch train loss 기록 
        train_loss = epoch_training_loss_sum / num_training
        epoch_loss.append(train_loss)

        # calculate time
        times = cal_run_time(time.time()-start)
        print('train_loss : {:.5f}\ttime : {}\n'.format(train_loss, times))

        # epoch 5 단위마다 model 저장
        if epoch%5==0:
            torch.save(model,f'eff_model_{epoch}.pt')


        '''
        ## Validation
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
  
        mlb = MultiLabelBinarizer(classes=['0','1','2','3','4','5','6','7','8','9','10','11'])
        mlb.fit(map(str.split, validation_true_labels_df['labels'].values))

        macro_f1_validation = f1_score(mlb.transform(map(str.split, validation_true_labels_df['labels'].values)), mlb.transform(map(str.split, validation_prediction_df['labels'].values)), average='macro')
        epoch_score.append(macro_f1_validation)

        # calculate time
        times = cal_run_time(time.time()-start())

        print('train_loss : {:.5f}\tvalid_f1_score : {:.5f}\ttime : {}\n'.format(train_loss, macro_f1_validation, times))


        # deep copy the model & save best model
        if macro_f1_validation > best_f1_score:
            best_idx = epoch
            best_f1_score = macro_f1_validation
            #best_model_wts = copy.deepcopy(model.state_dict())
            #torch.save(model.state_dict(), f'eff2_best_model.pt')
            torch.save(model, f'eff2_best_model.pt')
            print('==> best model saved - epoch: %d / f1_score: %.5f\n'%(best_idx, best_f1_score))
        

    times = cal_run_time(time.time()-train_start())
    print('Training complete in {}'.format(times))
    print('Best valid f1 score: %d - %.5f' %(best_idx, best_f1_score))
    '''

    # calculate time
    times = cal_run_time(time.time()-train_start)
    print('Training completed in {}'.format(times))

    # save epoch_loss & epoch_score
    with open('epoch_data', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(epoch_loss)
        #writer.writerow(epoch_score)
    print('\nepoch data saved.')

    #return model, best_idx, best_f1_score, epoch_loss, epoch_score, validation_prediction_df, validation_true_labels_df
    
    # Train start
train_model(model, criterion, optimizer, scheduler_poly_lr_decay, EPOCHS)

