import json
import matplotlib.pyplot as plt
import numpy as np
import sys
USAGE = "python plot_CV_results.py fname n_time_points"

plt.rc('font', size=16)
if(len(sys.argv) > 1):
    args = sys.argv
    try:
        f_name = args[1]
        file= open(f_name,'r')
        n_time_points = int(args[2])
    except Exception as e:
        print (USAGE)
        exit()

else:
    n_time_points = 50
    file= open('./cv_'+str(n_time_points)+'t.txt','r')
path = "./"

lines = json.load(file)
n_models = len(lines)/2
print("There are "+str(n_models)+" models to plot in this file.") #should be 4

if(n_models<4):
    print("Print the models all together")
    colors = ['r','r','b','b','g','g']
    fig1 = plt.figure(figsize=(8,8))
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Cross entropy)')
    for i in range(0,len(lines),2):
        model = lines[i][0]
        tr_loss = lines[i][2][2:]
        val_loss = lines[i+1][2][2:]
        epochs = range(2,len(tr_loss)+2)
        plt.plot(epochs,tr_loss,label=model+', train',color=colors[i],linestyle='--')
        plt.plot(epochs,val_loss,label=model+', validation',color=colors[i+1])
    plt.legend()
    plt.savefig(path+"loss_all_"+str(n_time_points)+".png")
    fig2 = plt.figure(figsize=(8,8))
    plt.xlabel('Epoch')
    plt.ylabel('Classification error rate')
    for i in range(0,len(lines),2):
        model = lines[i][0]
        tr_err = lines[i][3][2:]
        val_err = lines[i+1][3][2:]
        epochs = range(2,len(tr_err)+2)
        plt.plot(epochs,tr_err,label=model+', train',color=colors[i],linestyle='--')
        plt.plot(epochs,val_err,label=model+', validation',color=colors[i+1])
        plt.ylim((0,0.7))
    plt.legend()
    plt.savefig(path+"classErr_all_"+str(n_time_points)+".png")
else:
    print("Prints the models reagrouping the last two together and the rest together.")
    colors = ['r','r','b','b','g','g','r','r','g','g']
    n_last = 2
    fig1 = plt.figure(figsize=(8,8))
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Cross entropy)')
    for i in range(0,len(lines)-2*n_last,2):
        print(lines[i])
        model = lines[i][0]
        tr_loss = lines[i][2][2:]
        val_loss = lines[i+1][2][2:]
        epochs = range(2,len(tr_loss)+2)
        plt.plot(epochs,tr_loss,label=model+', train',color=colors[i],linestyle='--')
        plt.plot(epochs,val_loss,label=model+', validation',color=colors[i+1])
    plt.legend()
    plt.savefig(path+"loss_baselines_"+str(n_time_points)+".png")
    fig2 = plt.figure(figsize=(8,8))
    plt.xlabel('Epoch')
    plt.ylabel('Classification error rate')
    for i in range(0,len(lines)-2*n_last,2):
        model = lines[i][0]
        tr_err = lines[i][3][2:]
        val_err = lines[i+1][3][2:]
        epochs = range(2,len(tr_err)+2)
        plt.plot(epochs,tr_err,label=model+', train',color=colors[i],linestyle='--')
        plt.plot(epochs,val_err,label=model+', validation',color=colors[i+1])
        plt.ylim((0,0.7))
    plt.legend()
    plt.savefig(path+"classErr_baselines_"+str(n_time_points)+".png")

    fig3 = plt.figure(figsize=(8,8))
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Cross entropy)')
    for i in range(len(lines)-2*n_last,len(lines),2):
        model = lines[i][0]
        tr_loss = lines[i][2][2:]
        val_loss = lines[i+1][2][2:]
        epochs = range(2,len(tr_loss)+2)

        plt.plot(epochs,tr_loss,label=model+', train',color=colors[i],linestyle='--')
        plt.plot(epochs,val_loss,label=model+', validation',color=colors[i+1])
    plt.legend()
    plt.savefig(path+"loss_deeper_"+str(n_time_points)+".png")

    fig4 = plt.figure(figsize=(8,8))
    plt.xlabel('Epoch')
    plt.ylabel('Classification error rate')
    for i in range(len(lines)-2*n_last,len(lines),2):
        model = lines[i][0]
        tr_err = lines[i][3][2:]
        val_err = lines[i+1][3][2:]
        epochs = range(2,len(tr_err)+2)

        plt.plot(epochs,tr_err,label=model+', train',color=colors[i],linestyle='--')
        plt.plot(epochs,val_err,label=model+', validation',color=colors[i+1])

    plt.legend()
    plt.savefig(path+"classErr_deeper_"+str(n_time_points)+".png")
