#!/usr/bin/python
###############################################################################
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os
import re
import sys
from module_neat import *
os.system("rm plot_*")
#--------------------------------
# GA Settings:
#--------------------------------
npop=150 
itmax=600

weightcost=0.00002
niching_treshold=2.5
#==============================================
# Get Data
#==============================================
def fun_readdatafile(filename):
    DATA=np.loadtxt(filename)
    pattern=re.compile(r'#\s*p(\d+)\s*r(\d+)\s*')
    f1=open(filename,"r")
    for line in f1:
        m=pattern.match(line[:-1])
        if m:
            ninput=int(m.group(1))
            noutput=int(m.group(2))
            break
    f1.close()
    
    Xd,Yd=DATA[:,:ninput],DATA[:,ninput:]
    
    return [Xd,Yd]
#--------------------------------


[Xd,Yd]=fun_readdatafile("data_branin.dat")
ymax=np.max(Yd,axis=0)
ymin=np.min(Yd,axis=0)
xmax=np.max(Xd,axis=0)
xmin=np.min(Xd,axis=0)

ymin,ymax = np.asarray([-5.0]),np.asarray([210])


Y=(Yd-ymin)/(ymax-ymin)
X=(Xd-xmin)/(xmax-xmin)


#==============================================
# Initialize Neural Network
#==============================================

NN=neural_network(npop,X.shape[1],Y.shape[1],weightcost=weightcost,nodecost=0.0,connectioncost=0.0,sigma_tres=niching_treshold,maxhiddennnodes=5)


#==============================================
# GA Training
#============================================== 
#--------------------------------------------
dnas=NN.initialization(bias_prob=0.2,con_prob = 0.5)
NN.initialize_optimization(X,Y)
#--------------------------------------------


for it in range(itmax):
    
    dnas = NN.fun_evolve(dnas,mutate_con=0.08,mutate_node=0.02,mutate_w=0.9,mutate_bias=0.25)
    [fitness_ave,fitness_best,popsize]=NN.fun_print_ave_fitness(it)
    NN.fitnesslog()
    
NN.fitnesslog()
NN.plots2gif()

#==============================================
# PLOT TARGETS
#==============================================

Ydhat=ymin+(ymax-ymin)*NN.forward_propagation(NN.return_best_member(),X)

for n in range(Yd.shape[1]):
    plt.plot([ymin[n],ymax[n]],[ymin[n],ymax[n]],'k--',lw=3)

    plt.plot(Yd[:,n],Ydhat[:,n],'bo')

    plt.grid(True)
    plt.xlabel("Data")
    plt.ylabel("Prediction")
              
    plt.savefig("plot_target_"+str(n)+".png", dpi=None, facecolor='w', edgecolor='w',
    orientation='portrait', papertype=None, format="png",
    transparent=False, bbox_inches='tight', pad_inches=0.1,
    frameon=None)
    plt.close()

