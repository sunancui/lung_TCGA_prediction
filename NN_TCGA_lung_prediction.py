import numpy as np
import pandas as pd
import os
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
# processing the data, convert to numeric values
def data_processing(data_df):
    data_df['gender']=data_df['gender'].replace(["MALE","FEMALE"],[0,1])
    data_df['pathologic_N']=data_df['pathologic_N'].replace(["N0","N1","N2","N3","NX"],[0,1,2,3,np.nan])
    data_df['pathologic_stage']=data_df['pathologic_stage'].replace(["Stage IA","Stage IB","Stage IIA","Stage IIB","Stage IIIA","Stage IIIB","[Discrepancy]"],[1,1,2,2,3,3,np.nan])
    data_df['pathologic_T']=data_df['pathologic_T'].replace(["T1","T1a","T1b","T2","T2a","T2b","T3","T4"],[1,1,1,2,2,2,3,4])
    data_df['other_dx']=data_df['other_dx'].replace(["No","Yes, History of Prior Malignancy"],[0,1])
    data_df['tobacco_smoking_history']=data_df['tobacco_smoking_history'].replace(["1","2","3","4","[Not Available]"],[1,2,3,4,np.nan])
    data_df['primary_outcome']=data_df['primary_outcome'].replace(["progressive",'local'],[1,0])
    #fill missing data with median values
    data_df_fill=data_df.fillna(data_df.median())
    return data_df_fill
os.chdir("/Users/sunan/Desktop/github/lung_TCGA_prediction")
AD_SC_patient=pd.read_csv("./rd_AD_SC.csv",index_col=0)
select_column=["gender","pathologic_N","pathologic_stage","pathologic_T", 'other_dx', 'tobacco_smoking_history','radiation_total_dose','primary_outcome']
data_all=AD_SC_patient.loc[:,select_column]
data_all_values=data_processing(data_all)
##features
data_X=data_all_values.iloc[:,:-1].values
scaler=StandardScaler()
scaler.fit(data_X)
X_scaled=scaler.transform(data_X)
##label
data_Y=data_all_values.iloc[:,-1].values
##train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X_scaled,data_Y,test_size=0.33,random_state=0,stratify=data_Y)
##convert numpy array to torch tensor
X_train_torch=torch.from_numpy(X_train)
Y_train_torch=torch.from_numpy(Y_train.reshape(-1,1))
X_test_torch=torch.from_numpy(X_test)
Y_test_torch=torch.from_numpy(Y_test.reshape(-1,1))
##set some parameters
input_dim=X_train.shape[1]
output_dim=1
hidden_dim=5
learning_rate=0.001
num_epoch=500
# define a 2-hidden layer fully-connected NN
class MLP_NN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim_1,hidden_dim_2, output_dim):
        super(MLP_NN, self).__init__()
        self.L1=torch.nn.Linear(input_dim,hidden_dim_1)
        self.D1=torch.nn.Dropout(0.2)
        self.L2=torch.nn.Linear(hidden_dim_1,hidden_dim_2)
        self.D2=torch.nn.Dropout(0.2)
        self.L3=torch.nn.Linear(hidden_dim_2,output_dim)
    def forward(self,x):
        a1=torch.relu(self.L1(x))
        a1=self.D1(a1)
        a2=torch.relu(self.L2(a1))
        a2=self.D2(a2)
        outputs=torch.sigmoid(self.L3(a2))
        return outputs
#initialize the model
model=MLP_NN(input_dim,hidden_dim,hidden_dim,output_dim)
#using binary cross entropy as loss
criterion=torch.nn.BCELoss(reduction='mean')
#using Adam optimizer
optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)
history_loss_train=[]
history_loss_test=[]
for i in range(num_epoch):
    #train data
    model.train()
    optimizer.zero_grad()
    y_pred=model(X_train_torch.float())
    #define the loss, adding l2 penalty
    loss=criterion(y_pred,Y_train_torch.float())
    loss.backward()
    optimizer.step()
    # evaluate on test data
    model.eval()
    y_pred_test=model(X_test_torch.float())
    loss_test=criterion(y_pred_test,Y_test_torch.float())
    if i%10==0:
        print(i, loss, loss_test)
    history_loss_test.append(loss_test.detach().numpy())
    history_loss_train.append(loss.detach().numpy())
# plot the history of training/test loss
plt.figure()
plt.plot(np.arange(0,num_epoch),history_loss_train,label="training")
plt.plot(np.arange(0,num_epoch),history_loss_test,label="validation")
plt.xlabel("epoch",fontsize=18)
plt.ylabel("Loss",fontsize=18)
plt.legend(prop={'size':16})
plt.show()
#calculate AUC for traing/test sets
auc_train=roc_auc_score(Y_train, y_pred.detach().numpy())
auc_test=roc_auc_score(Y_test, y_pred_test.detach().numpy())
fpr, tpr,thresholds=roc_curve(Y_train, y_pred.detach())
plt.figure()
fpr_t, tpr_t,thresholds=roc_curve(Y_test, y_pred_test.detach().numpy())
plt.plot(fpr,tpr,label="training")
plt.plot(fpr_t,tpr_t,label="test")
plt.legend(prop={'size':16})
plt.xlabel('1-Specificity',fontsize=16)
plt.ylabel('Sensitivity',fontsize=16)
plt.show()