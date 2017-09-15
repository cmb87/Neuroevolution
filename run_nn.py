#!/usr/bin/python
####################################################################################################################
import numpy as np
import math
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D
import module_neft as ne
import time
import re
import os
os.system("rm plot_*")
#==============================================
# Settings
#==============================================
trainwga=True
trainwgd=False

weightcost=0.00001
#--------------------------------
# GA Settings:
#--------------------------------
npop=150 
itmax=1000
niching_treshold=1.6
#-------------------------------- 

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
# Initialize Neural Networks
#==============================================
NN_gd=ne.neural_network(topology=[Xd.shape[1],3,Yd.shape[1]],usebias=False,conwcost=weightcost)   # Training by gradient descent
NN_ga=ne.neural_network(topology=[Xd.shape[1],5,Yd.shape[1]],usebias=True,conwcost=weightcost)   # Training by GA

[nnodes,ncons,nbias,wmax]=NN_ga.get_network_props()

#==============================================
# GA Training
#============================================== 
if trainwga:
    #-------------------------------- 
    def fun_wrapper(wvec):
        NN_ga.vec2weights(wvec)
        NN_ga.feed_forward(X)
        NN_ga.calc_error(Y)    
        return NN_ga.return_cost()
    #-------------------------------- 
    
    xlo=-wmax*np.ones(ncons+nbias)
    xup= wmax*np.ones(ncons+nbias)
        
    ga=ne.genetic_algorithm(fun_wrapper,xlo,xup,niching_treshold=niching_treshold)
    
    #--------------------------------
    # Initial Population
    #--------------------------------
    pop = np.zeros((npop,ncons+nbias))
    pop[:,:ncons] = (-wmax + (2*wmax)*np.random.rand(npop,ncons))

    #--------------------------------
    for it in range(itmax):
        
        pop=ga.fun_evolve(pop,mutate=0.15,randomcrossover=True,no_childs=2)
        
        [fitness_ave,fitness_best,popsize]=ga.fun_print_ave_fitness(it)
        #NN_ga.vec2weights(ga.fun_best_member())
        #NN_ga.shownetwork(plotname="plot_neural_network_iter_"+str(it)+".png")
        
    ga.fitnesslog()
    NN_ga.vec2weights(ga.fun_best_member())
    NN_ga.shownetwork(plotname="plot_neural_network_iter_"+str(it)+".png")
    
    NN_ga.vec2weights(ga.fun_best_member())

#==============================================
# Gradient Descent Training
#==============================================
if trainwgd:
   # NN_gd.vec2weights(ga.fun_best_member())
    NN_gd.train(Xd,yd)
    NN_gd.shownetwork()
    vec_gd=NN_gd.weights2vec()

#==============================================
# PLOT TARGETS
#==============================================

if trainwgd:
   Ydhat_gd=ymin+(ymax-ymin)*NN_gd.feed_forward(X)
if trainwga:   
   Ydhat_ga=ymin+(ymax-ymin)*NN_ga.feed_forward(X)

for n in range(Yd.shape[1]):
    plt.plot([ymin[n],ymax[n]],[ymin[n],ymax[n]],'k--',lw=3)

    if trainwgd:
        plt.plot(Yd[:,n],Ydhat_gd[:,n],'bo')
    if trainwga:    
        plt.plot(Yd[:,n],Ydhat_ga[:,n],'bo')

    plt.grid(True)
    plt.xlabel("Data")
    plt.ylabel("Prediction")
              
    plt.savefig("plot_target_"+str(n)+".png", dpi=None, facecolor='w', edgecolor='w',
    orientation='portrait', papertype=None, format="png",
    transparent=False, bbox_inches='tight', pad_inches=0.1,
    frameon=None)
    plt.close()
