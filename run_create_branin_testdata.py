#!/usr/bin/python
####################################################################################################################
import numpy as np
import math
#==============================================
Nsmpls=40
xlo=np.asarray([-5,0])
xup=np.asarray([10,15])
#==============================================
def fun_test_branin(a):
    x,y=a[:,0],a[:,1]
    z=np.zeros((a.shape[0],1))
    a,b,c=1,5.1/(4*np.pi**2),5/np.pi
    r,s,t=6,10,1/(8*np.pi)
    z[:,0]=a*(y-b*x**2+c*x-r)**2+s*(1-t)*np.cos(x)+s

    return z
#==============================================


X=xlo+(xup-xlo)*np.random.rand(Nsmpls,2)
y=fun_test_branin(X)

DATA=np.zeros((Nsmpls,3))
DATA[:,:2]=X[:]
DATA[:,2]=y[:,0]

f1=open("data_branin.dat",'w')
f1.write("# Branin function test data sample\n")
f1.write("# p2 r1\n")
np.savetxt(f1,DATA)
f1.close()
