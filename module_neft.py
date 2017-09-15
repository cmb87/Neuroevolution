#!/usr/bin/python
############################################################################################################
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os
import re

np.set_printoptions(precision=4)
#====================================================
# GENETIC ALGORITHM
#====================================================
class genetic_algorithm(object):

    # =======================================================
    def __init__(self,fct,xlo,xup,niching_treshold=1.5):
        self.fct=fct
        self.xlo=xlo
        self.xup=xup
        self.fitness_best=0
        self.fitness_worst=0
        self.fitness_ave=0
        self.fitnesslog_best=[]
        self.fitnesslog_worst=[]
        self.fitnesslog_ave=[]
        self.sigma_tres=niching_treshold
    # =======================================================
    def fun_in_designspace(self,x):
        val=np.all(self.xup>=x) and np.all(x>=self.xlo)
        return val
    # =======================================================
    def fun_mutate_gene(self,array):
        mutateindex=np.random.randint(len(array))
        array[mutateindex]=self.xlo[mutateindex]+np.random.rand()*(self.xup[mutateindex]-self.xlo[mutateindex])
        return array

    # =======================================================
    def fun_eval_fitness(self,x,callbackfct=None):
            return self.fct(x)
            
    # =======================================================
    # =======================================================
    # Evolution Operator
    # =======================================================
    # =======================================================
    def fun_evolve(self,pop,mutate=0.05,no_childs=2,randomcrossover=False):
    
        nvars,pop_size=pop.shape[1],pop.shape[0]
        self.pop_size=pop_size
        #----------------
        sh=self.niching_sharefunction(pop)
        
        #----------------
        fitness=np.asarray([self.fun_eval_fitness(pop[x,:]) for x in range(pop_size)])
        fitness_adjusted=fitness[:]*np.sum(sh,axis=0)
        
        
        #print np.sum(sh,axis=0)
        ranking=np.argsort(fitness)
        ranking_adjusted=np.argsort(fitness_adjusted)
        #-------------------------------------
        self.fitness_ave= np.mean(fitness)
        self.fitness_best = fitness[ranking[0]]
        self.fitness_worst = fitness[ranking[-1]]

        self.fitnesslog_best.append(self.fitness_best)
        self.fitnesslog_worst.append(self.fitness_worst)
        self.fitnesslog_ave.append(self.fitness_ave)
        #-------------------------------------
        proba_ranked=self.get_probability_list(pop_size)

        rankedpop=pop[ranking_adjusted,:].tolist()

        self.bestmember=np.asarray(pop[ranking[0]])

        #-------------------------------------
        # Elitists
        #-------------------------------------
        pop_next = [pop[ranking[0]]]

        #-------------------------------------
        # Crossover
        #-------------------------------------
        desired_length = pop_size - len(pop_next)
        children = []
        #=========          
        while len(children) < desired_length:
            [male_index,female_index]=self.roulette_wheel_pop(proba_ranked,2)
            
            if sh[male_index,female_index] == 1  and male_index != female_index:
                
                male=rankedpop[male_index]
                female=rankedpop[female_index]
                
                if randomcrossover:
                    coi = nvars/2 + np.random.randint(-nvars/2,nvars/2)
                else:
                    coi = nvars/2 
                #=========
                if no_childs == 1 :
                    child1 = female[:coi] + male[coi:]
                    children.append(child1)
                elif no_childs == 2 :
                    child1 = female[:coi] + male[coi:]
                    child2 = male[:coi] + female[coi:]
                    children.append(child1)
                    children.append(child2)
        #=========          
        pop_next.extend(children[:desired_length])
        #=========          

        #-------------------------------------
        # Mutation
        #-------------------------------------
        # mutate some individuals
        for n in range(1,len(pop_next)):
            if mutate > np.random.rand():
                #pop_next[n]=self.fun_mutate(individual)
                pop_next[n]=self.fun_mutate_gene(pop_next[n])

        #-------------------------------------
        # Return new population
        #-------------------------------------

        return np.asarray(pop_next)
        
    # =======================================================
    # =======================================================
    # Share Function
    # =======================================================
    # =======================================================
    def niching_sharefunction(self,pop):
        sh=np.zeros((len(pop),len(pop)))
        for i in range(len(pop)):
            sh[i,i]=1
            for j in range(len(pop)):
                if i<j:
                    sigma=np.abs(np.mean(pop[i])-np.mean(pop[j]))
                    #print sigma
                    if sigma < self.sigma_tres:
                        sh[i,j],sh[j,i]=1,1
        return sh
    # =======================================================
    # =======================================================
    # Roulette Wheel
    # =======================================================
    # =======================================================
    def get_probability_list(self,npop):
    
        aux=1.0/np.arange(1,npop+1)
        ranks=(aux)/np.sum(aux)

        probabilities = [np.sum(ranks[:i+1]) for i in range(npop)]

        return probabilities
    # =======================================================
    def roulette_wheel_pop(self,probabilities,nselect):
        #-------------------------------------
        chosen = []
        while len(chosen)<nselect:
            r = np.random.random()
            for i in range(len(probabilities)):
                if r <= probabilities[i] and not i in chosen:
                    chosen.append(i)
                    break
        return sorted(chosen)
    # =======================================================
    # =======================================================
    # GET PARETO RANKING
    # =======================================================
    # =======================================================
    def calc_pareto_rank(self,Y):
        #------------------------------------------------
        def dominates_check(row, rowCandidate):
            return all(r <= rc for r, rc in zip(row, rowCandidate))

        def cull(pts,pts_index, dominates):
            dominated = []
            dominated_index = []
            cleared = []
            cleared_index = []
            remaining = pts
            remaining_index = pts_index
            
            while remaining:
                candidate = remaining[0]
                candidate_index = remaining_index[0]
                new_remaining = []
                new_remaining_index = []
                
                for other,other_index in zip(remaining[1:],remaining_index[1:]):
                    [new_remaining, dominated][dominates(candidate, other)].append(other)
                    [new_remaining_index, dominated_index][dominates(candidate, other)].append(other_index)
                    
                if not any(dominates(other, candidate) for other in new_remaining):
                    cleared.append(candidate)
                    cleared_index.append(candidate_index)
                else:
                    dominated.append(candidate)
                    dominated_index.append(candidate_index)
                    
                remaining = new_remaining
                remaining_index = new_remaining_index
            return cleared_index, dominated,dominated_index
        #------------------------------------------------
        dominated=Y[:].tolist()
        dominated_index=range(Y.shape[0])
        rank=0
        paretoranks=Y.shape[0]*[""]
        
        while not dominated == []:
            [rankedpts, dominated,dominated_index]= cull(dominated,dominated_index, dominates_check)
            for rankedpt in rankedpts:
                paretoranks[rankedpt]=rank
            rank+=1
        #------------------------------------------------
        return paretoranks
    # =======================================================
    # =======================================================
    def fitnesslog(self):
        plt.semilogy(self.fitnesslog_ave,'k-')
        plt.plot(self.fitnesslog_worst,'r-')
        plt.semilogy(self.fitnesslog_best,'b-', lw=2)
        
        plt.xlabel("Iterations")
        plt.ylabel("Fitness")
        
        plt.grid(True)

        plt.savefig("plot_convergence_iter.png", dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format="png",
        transparent=False, bbox_inches='tight', pad_inches=0.1,
        frameon=None)
        plt.close()
    # =======================================================
    def fun_print_ave_fitness(self,n):
    
        ave = '%.3e' % self.fitness_ave
        best = '%.3e' % self.fitness_best
        worst = '%.3e' % self.fitness_worst
    
        print  str(n)+"\t"+str(ave)+"\t"+str(best)+"\t"+str(worst)
        return [self.fitness_ave,self.fitness_best,self.pop_size]

    # =======================================================
    def fun_best_member(self):
        return np.asarray(self.bestmember)
        
#====================================================
#====================================================
# FIXED TOPOLOGIE NEURAL NETWORK (NEFT)
#====================================================
#====================================================
class neural_network(object):
    # =======================================================
    def __init__(self,topology=[2,6,5,1],layertype=["c","s","s","s"],usebias=False,conwcost=0.1):
        self.topology=topology
        self.nolayer=len(topology)
        self.usebias=usebias
        self.learningrate=0.8
        self.wmax=16.0
        self.nnodes=sum(self.topology)
        self.ncons=0
        self.nbias=0
        self.layertype=layertype
        self.weightcost=conwcost
        self.bias=[]
        # Initialize weights
        np.random.seed(50)
        W=self.nolayer*['']
        for layer in range(self.nolayer-1):
            W[layer]=(2*np.random.random((topology[layer],topology[layer+1])) - 1)
            self.ncons+=topology[layer]*topology[layer+1]
            self.nbias+=topology[layer+1]
            self.bias.append(np.zeros(topology[layer+1]))
        self.W=W

    # =======================================================
    def get_network_props(self):
        if self.usebias:
            return[self.nnodes,self.ncons,self.nbias,self.wmax]
        else:
            return[self.nnodes,self.ncons,0,self.wmax]
    # =======================================================
    def shownetwork(self,plotname="plot_neural_network.png"):
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        
        for layer in range(self.nolayer-1):
            inl,rnl = self.W[layer].shape[0],self.W[layer].shape[1]
            
            x=[layer+0.2,layer+1+0.2]
            
            for r in range(rnl):
                for i in range(inl):
                    y=[-0.5*inl+i+0.2,-0.5*rnl+r+0.2]
                    if self.W[layer][i,r]>0:
                        plt.plot(x,y,'r-',lw=(np.abs(self.W[layer][i,r])+1)/(self.wmax+1))
                    if self.W[layer][i,r]<0:
                        plt.plot(x,y,'b-',lw=(np.abs(self.W[layer][i,r])+1)/(self.wmax+1))
                    
                    plt.plot(x,y,'ko',markersize=10)
                                          
                if self.bias[layer][r] > 0.0:
                    plt.plot([x[1]-0.20,x[1]],[y[1]-0.25,y[1]],'r-',lw=3*(np.abs(self.bias[layer][r])+1)/(self.wmax+1))
                    plt.plot([x[1]-0.20],[y[1]-0.25],'k^',markersize=8)
                elif self.bias[layer][r] < 0.0:
                    plt.plot([x[1]-0.20,x[1]],[y[1]-0.25,y[1]],'b-',lw=3*(np.abs(self.bias[layer][r])+1)/(self.wmax+1))
                    plt.plot([x[1]-0.20],[y[1]-0.25],'k^',markersize=8)

        #string="Specie: "+str(dna["name"])+"\n"
        string=""
        string=string+"Number of nodes: "+str(self.nnodes)+"\n"
        string=string+"Number of connections: "+str(self.ncons)+"\n"
        #string=string+"Generation:  "+str(dna["generation"])+"\n"
        #string=string+"Parents:   "+str(dna["parents"][0])+" x "+str(dna["parents"][1])+"\n"
        #string=string+"Survived:   "+str(dna["survived"])+"\n"

        plt.annotate(string, (0.8,0), (0, 25), xycoords='axes fraction', textcoords='offset points', fontsize=8,ha='center', va='center', bbox=props)
        plt.axis('off')

        plt.axis('off')            
        plt.savefig(plotname, dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format="png",
        transparent=False, bbox_inches='tight', pad_inches=0.1,
        frameon=None)
        plt.close()
       
    # =======================================================
    def train(self,Xt,yt,iterback=60000):
        nplot=0
        nplotmax=10000
        nplotstop=10000
        plotnames=""
        for n in range(iterback):

            self.feed_forward(Xt)

            if nplot == nplotmax and nplotstop>n:
                print "Plot "+str(n)
                plotname="plot_neural_network_"+str(n)+".png"
                plotnames=plotnames+" "+plotname
                self.shownetwork(plotname=plotname)
                nplot=0
                
            error=self.calc_error(yt)
            self.feed_backward()
            print str(n)+"\t"+str(self.return_cost())
            nplot+=1
        #os.system("convert "+plotnames+" plot_simulation.gif")
        os.system("rm -f "+plotnames)

    # =======================================================
    def weights2vec(self):
        vec=[]
        for layer in range(self.nolayer-1):
            imax,jmax=self.W[layer].shape[0],self.W[layer].shape[1]
            vec.extend(self.W[layer].reshape(imax*jmax).tolist())
        if self.usebias:
            for layer in range(self.nolayer-1):
                vec.extend(self.bias[layer].tolist())

        return np.asarray(vec)

    # =======================================================
    def vec2weights(self,vec):
        sstart=0
        for layer in range(self.nolayer-1):
            imax,jmax=self.W[layer].shape[0],self.W[layer].shape[1]
            self.W[layer] = vec[sstart:sstart+(imax*jmax)].reshape((imax,jmax))
            sstart+=imax*jmax
        if self.usebias:
            vecbias=vec[self.ncons:]
            sstart=0
            for layer in range(1,self.nolayer):
                layernodesize=self.topology[layer]
                self.bias[layer-1]=np.asarray(vecbias[sstart:sstart+layernodesize])
                sstart+=layernodesize
    # =======================================================
    def sigmoid(self,x,deriv=False):
        if(deriv==True):
            return x*(1-x)
        return 1/(1+np.exp(-x))

    # =======================================================
    def feed_forward(self,X):
        layer_rspns=self.nolayer*[0]
        layer_rspns[0]=X
        for layer in range(1,self.nolayer):
            if self.layertype[layer] == "s":
                layer_rspns[layer]=self.sigmoid(self.bias[layer-1]+np.dot(layer_rspns[layer-1],self.W[layer-1]))
            elif self.layertype[layer] == "c":
                layer_rspns[layer]=0.1*(self.bias[layer-1]+np.dot(layer_rspns[layer-1],self.W[layer-1]))
                
        self.layer_rspns=layer_rspns

        return layer_rspns[-1]

    # =======================================================
    def calc_error(self,y):
    
        self.error=(y-self.layer_rspns[-1])     
        cost_error = np.mean(np.abs(self.error)**2)
        
        cost_weights=0.0
        for layer in range(self.nolayer-1):
            cost_weights+= 0.5*self.weightcost/self.ncons*np.sum(np.abs(self.W[layer]**2))

        if cost_error/(cost_weights+1e-20) < 10:
            print "WARNING! Structure cost are below one magnitude big than error!"

        self.cost= cost_error + cost_weights

        return np.mean(np.abs(self.error))

    # =======================================================

    def return_cost(self):
        return self.cost
    # =======================================================
    def feed_backward(self):
        # in what direction is the target value?
        # were we really sure? if so, don't change too much.

        layer_delta=self.nolayer*[0]
        layer_error=self.nolayer*[0]
        layer_error[-1]=self.error
        
        for layer in range(self.nolayer-1,1,-1):
            layer_delta[layer] = layer_error[layer]*self.sigmoid(self.layer_rspns[layer],deriv=True)
            layer_error[layer-1] = np.dot(layer_delta[layer],self.W[layer-1].T)
            self.W[layer-1]+=self.learningrate*np.dot(self.layer_rspns[layer-1].T,layer_delta[layer])

#------------------------------------------------------------

