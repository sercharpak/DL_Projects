import sys, getopt
import matplotlib.pyplot as plt
"""Execution options :
    one_khz : True to use data with the highest sampling rate (500 points), vs 50 points
    cross_val : Return cross-validation results as well"""
if(len(sys.argv) > 1):
    args = sys.argv
    if(args[1] == '1'):
        one_khz = True
    else:
        one_khz = False

    if(args[2] == '1'):
        cross_val = True
    else:
        cross_val = False

else:
    #Default parameters
    one_khz = True
    cross_val = False

"""Load the data and standardize"""
import dlc_bci as bci
import torch

dtype = torch.float
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Loading the data and standardizing...")
train_input ,train_target = bci.load( root = './data_bci',one_khz=one_khz)
test_input,test_target = bci.load(root='./data_bci', train = False, one_khz=one_khz)
test_input = (test_input - train_input.mean(0))/train_input.std(0)
train_input = (train_input-train_input.mean(0))/train_input.std(0)


"""Import the models"""
from models import *
print("Initializing model list...")
if(one_khz) :
    n_time_points = 500
    sampling_rate = 1000
else:
    n_time_points = 50
    sampling_rate = 100
n_channels = 28

model_list = [  NHLP(n_time_points*n_channels),
                SimpleConv(n_channels,n_time_points,int(n_time_points/10)+1,3),
                EEGNet_2018(n_channels,n_time_points),
                ShallowConvNet(n_channels,n_time_points)
                ]
optimizer_list = [torch.optim.Adam]*len(model_list)
scheduler_gamma_list = [0.99,0.99,0.995,0.995]
criterion_list = [nn.CrossEntropyLoss()]*len(model_list)
if(n_time_points == 50):
    lr_list = [2e-1,2e-1,1e-2,2e-1]
    #lr_list = [1e-1]*len(model_list)
elif(n_time_points == 500):
    lr_list = [1e-3,1e-3,1e-2,1e-3]
epochs_list = [75,75,75,150]



import cross_validation as cv #Also used for the train_model, even without cross validation_data
import json

dump_final = []
if(cross_val):
    dump = []

for i in range(len(model_list)):
    model = model_list[i]
    optimizer = optimizer_list[i](model.parameters(),lr=lr_list[i])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,scheduler_gamma_list[i])
    criterion = criterion_list[i]
    print("Model {} of {} : {}".format(i+1,len(model_list),model.name()))
    if(cross_val):
        k_fold = 4
        print("Cross validating... on {} folds".format(k_fold))
        tr_loss, val_loss, tr_err, val_err = cv.cross_validate(model, criterion,optimizer,scheduler,train_input,train_target,k_fold,batch_size=10,n_epochs=epochs_list[i],n_augmentation=0,verbose=2)
        print("Mean train error : {}, mean validation error : {}".format(tr_err[-1],val_err[-1]))
        dump.append((model.name(), " train ", tr_loss, tr_err))
        dump.append((model.name(), " validation ", val_loss, val_err))

    model.reset()
    print("Training...")
    cv.train_model(model,criterion,optimizer,scheduler,train_input,train_target,n_epochs=epochs_list[i],batch_size=10,n_augmentation=0,verbose=2)
    final_tr_error = cv.evaluate_error(model,train_input,train_target)
    final_te_error = cv.evaluate_error(model,test_input,test_target)
    print("Train error = {} ; Test error = {} ".format(final_tr_error,final_te_error))

    dump_final.append((model.name(), " train ", final_tr_error.item()))
    dump_final.append((model.name(), " test " , final_te_error.item()))


file = open('final_'+str(n_time_points)+'t.txt','w+')
json.dump(dump_final,file)
file.close()

if(cross_val):
    file = open('cv_'+str(n_time_points)+'t.txt','w+')
    json.dump(dump,file)
    file.close()
