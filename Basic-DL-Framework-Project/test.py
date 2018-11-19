import torch
from torch import Tensor
import math
import matplotlib.pyplot as plt
import json

import module as m
import loss as l
import optim as o
from utils import convert_to_one_hot_labels, generate_disk_dataset, plot_with_labels

"""This version of the code represent the labels as one-hot-labels, meaning the models should have
output of dimension 2"""

# Optional : fix the seed
#torch.manual_seed()

# Generate train set and test set
print("Generating dataset...")
nPoints = 1000
train, train_label = generate_disk_dataset(nPoints)
test, test_label = generate_disk_dataset(nPoints)

# Select model
print("Building the model...")
model = m.Sequential(m.Linear(2,25), m.ReLU(), m.Linear(25,25), m.ReLU(), m.Linear(25,25), m.ReLU(), m.Linear(25,2))
#model = m.Sequential(m.Linear(2,25), m.ReLU(), m.Linear(25,2, bias = False))
#model = m.Sequential(m.Linear(2,128), m.ReLU(), m.Linear(128,2))
#model = m.Sequential(m.Linear(2,2))

# #Select optimizer
# optim = o.GradientDescent(0.04)
optim = o.SGD(10,0.01)
# optim = o.SGDWithRepetition(153,0.05)

# #Select loss
loss = l.MSE()
# loss = l.CrossEntropyLoss()

# #Train the model and plot the train loss evolution
print("Training the model...")
v = optim.train_model(model, train, train_label, loss, n_epochs = 100)
file = open('train_loss.out','w')
json.dump(v,file)
file.close()

fig = plt.figure()
plt.plot(v[2:]) # Do not plot initial loss that strictly depends on the weights initialization
plt.title("Training loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

# Evaluate the error on the test set
print("Evaluating model on the test set...")
output = model.forward(test)
print('Test loss = {}'.format(loss.loss(output,test_label)))
print('Missclassified points = {} / {}'.format(l.ClassificationLoss.loss(output, test_label)[0], nPoints))


# #Plot the predictions and the ill classified points
train_label = train_label.max(dim=0)[1].long()
test_label = test_label.max(dim=0)[1].long()
train_pred = model.forward(train).max(dim=0)[1].long().squeeze()
test_pred =  output.max(dim=0)[1].long().squeeze()
train_miss = (torch.abs(train_pred.float()-train_label.float())!=0).long()
test_miss = (torch.abs(test_pred.float()-test_label.float())!=0).long()


""" PLOTTING """
plt.subplots_adjust(hspace=0.2)
plt.margins(0,0)
# Plots for train
fig_train, axes = plt.subplots(nrows=3,ncols=1,figsize=(4,14),sharex=True)
plot_with_labels(train,train_label,axes[0])
axes[0].set_title("Train labels")

plot_with_labels(train,train_pred,axes[1])
axes[1].set_title("Train predictions")

plot_with_labels(train,train_miss,axes[2])
axes[2].set_title("Train missclassification")

# Plots for test
fig_test, axes = plt.subplots(nrows=3,ncols=1,figsize=(4,14),sharex=True)
plot_with_labels(test,test_label,axes[0])
axes[0].set_title("Test labels")

plot_with_labels(test,test_pred,axes[1])
axes[1].set_title("Test predictions")

plot_with_labels(test,test_miss,axes[2])
axes[2].set_title("Test missclassification")

plt.show()
