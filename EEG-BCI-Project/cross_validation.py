import torch
from torch import nn
from torch.autograd import Variable
from data_augmentation import augment_train
import math

def split_data(dataset,target,k_fold):
    """"
    Split a dataset in several subsets of equal size.
    Inputs : dataset -> original dataset
             target -> original target
             k_fold -> number of subsets required
    Outputs : all_datasets -> list of all subsets of the dataset
              all_targets -> list of subsets of the targets, matching with
                             all_datasets
    """
    N = dataset.size()[0]
    step = N//k_fold
    indices = torch.randperm(N);
    all_datasets = []
    all_targets = []
    for k in range(0,k_fold-1):
        all_datasets.append(dataset[indices[k*step:(k+1)*step]])
        all_targets.append(target[indices[k*step:(k+1)*step]])
    all_datasets.append(dataset[indices[(k_fold-1)*step:N]])
    all_targets.append(target[indices[(k_fold-1)*step:N]])
    return all_datasets, all_targets

def build_train_and_validation(all_datasets,all_targets,k_validation,k_fold):
    """
    Create a train set and a validation set from a list of subsets.
    Inputs : all_datasets -> list of datasets
             all_targets -> list of targets, matching with all_datasets
             k_validation -> index in the list of the subset that should be
                             used as the validation set
             k_fold -> number of subsets
    Outputs : train_data, train_target -> Train set
              validation_data, validation_target -> Validation set
    """
    assert((k_validation >= 0) & (k_validation < k_fold)),'Index of the validation set is out of bounds'
    train = list(range(0,k_validation))+list(range(k_validation + 1, k_fold))
    train_data = torch.cat([all_datasets[i] for i in train])
    train_target = torch.cat([all_targets[i] for i in train])
    validation_data = all_datasets[k_validation]
    validation_target = all_targets[k_validation]
    return train_data, train_target, validation_data, validation_target

def train_model(model, criterion, optimizer, scheduler,train_input, train_target,n_epochs=250,batch_size=50,n_augmentation =0,verbose=0,save_loss=False,tr_loss=[],val_loss=[],tr_err=[],val_err=[],val_input=torch.Tensor(),val_target=torch.Tensor()):
    """
    Trains the model, using criterion as the loss function, and the optimizer wrapped in scheduler
    Inputs: model -> the model to be trained (torch.nn.Module)
            criterion -> loss function to be used for optimization
            optimizer -> optimization algorithm
            scheduler -> scheduler, changing the learning rate throughout the epochs
            train_input -> input for training, size: (n_samples, n_channels, time_points)
            train_target -> trarget values, size: (n_samples)
            n_epochs -> number of passes through the whole dataset in training
            batch_size -> the batch size for training
            n_augmentation -> number of copies of the dataset with added noise that are to be appended to (train_input, train_target)
            verbose -> manages output: 1 - only informs about end of training, while 2 about the current epoch % 20
            save_loss -> a boolean indicating whether to evaluate the error& loss of the model on the whole training and validation sets
            + lists to store results, if save_loss == True
    """
    # Augment the data if necessary and make Variables
    train_input, train_target = augment_train(train_input, train_target, n_augmentation, 0.1, 2 , 0.2,verbose)
    train_input, train_target = Variable(train_input), Variable(train_target)
    if(save_loss):
        val_input,val_target = Variable(val_input),Variable(val_target)

    for e in range(n_epochs):
        scheduler.step() # decrease the learning rate
        if(verbose>=2):
            if(e % 20 == 19):
                print("Training epoch {} of {}".format(e+1,n_epochs),end='\r')

        # permutation of the samples, for batch training
        indices = torch.randperm(train_input.size(0))
        train_input = train_input[indices]
        train_target = train_target[indices]

        # train on batches
        for b in range(0,train_input.size(0),batch_size):
            if(train_input.size(0) - b >= batch_size):
                output = model(train_input.narrow(0,b,batch_size))
                loss = criterion(output,train_target.narrow(0,b,batch_size))
                model.zero_grad()
                loss.backward()
                optimizer.step()
        # save loss data, if save_loss is true
        if(save_loss):
            tr_loss.append(criterion(model(train_input),train_target).data[0])
            val_loss.append(criterion(model(val_input),val_target).data[0])
            tr_err.append(evaluate_error(model,train_input.data,train_target.data))
            val_err.append(evaluate_error(model,val_input.data,val_target.data))
    if(verbose == 1):
        print("Training ended successfully after {} epochs".format(n_epochs))

def evaluate_error(model, data_input, data_target):
    """
    Evaluates the classification error on data_input, with respect to data_target
    Inputs: model -> the model to evaluate performance (torch.nn.Module)
            data_input -> input data to evaluate over
            data_target -> the correct labels for the input data
    Outputs: the number of missclassified samples
    """
    data_input, data_target = Variable(data_input,volatile=True),Variable(data_target,volatile=True)
    nb_errors = 0

    output = model(data_input)
    nb_errors += (output.max(1)[1] != data_target).long().data.sum() # count the number of samples in the output with labels different than in the target

    return nb_errors/data_input.size(0) # take the mean, to get a fraction of missclassified samples

def cross_validate(model,criterion,optimizer,scheduler,dataset,target,k_fold,n_epochs=250,batch_size = 50,n_augmentation =0,verbose=0):
    """
    Given a model and a dataset, performs k-fold cross validation and return
    the list of the errors on each fold.
    Inputs : model, optimizer, scheduler, n_epochs, batch_size, n_augmentation -> look into train_model
             dataset, target -> dataset
             k_fold -> number of folds to use

    Outputs : train_errors -> list of the train error on each fold
              validation_errors -> list of the validation error on each fold
    """
    all_datasets, all_targets = split_data(dataset,target,k_fold) # prepare data for crossvalidation
    # containers for loss
    mean_tr_loss = [0 for i in range(n_epochs)]
    mean_val_loss = [0 for i in range(n_epochs)]
    mean_tr_err = [0 for i in range(n_epochs)]
    mean_val_err = [0 for i in range(n_epochs)]

    for k in range(0,k_fold):
        if(verbose >= 1):
            print('Cross-Validation : step {} of {}'.format(k+1,k_fold))

        tr_loss = []
        val_loss = []
        tr_err = []
        val_err = []


        model.reset() # reset the model parameters
        tr_data, tr_target, val_data,val_target = build_train_and_validation(all_datasets,all_targets,k,k_fold)
        train_model(model,criterion,optimizer,scheduler,tr_data,tr_target,n_epochs,batch_size,n_augmentation,verbose,True,tr_loss,val_loss,tr_err,val_err,val_data,val_target)
        mean_tr_loss = [old + new/k_fold for old,new in zip(mean_tr_loss,tr_loss)]
        mean_val_loss = [old + new/k_fold for old,new in zip(mean_val_loss,val_loss)]
        mean_tr_err = [old + new/k_fold for old,new in zip(mean_tr_err,tr_err)]
        mean_val_err = [old + new/k_fold for old,new in zip(mean_val_err,val_err)]
        temp_tr_err = evaluate_error(model,tr_data,tr_target)
        temp_val_err = evaluate_error(model,val_data,val_target)

        if(verbose >=2):
            print('End of step {} : Train error = {} , Validation error = {}'.format(k+1,tr_err[-1],val_err[-1]))


    return mean_tr_loss, mean_val_loss, mean_tr_err, mean_val_err
