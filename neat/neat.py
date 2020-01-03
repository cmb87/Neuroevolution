import numpy as np
import json
import sys
import os
import copy
import gym
import matplotlib.pyplot as plt
import multiprocessing
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))

from src.neat.genom import Genom

class NEAT:
    ### Constructor ###
    def __init__(self, xdim, ydim, npop=100, maxtimelevel=1, output_activation=None, nparallel=3):
        self.currentIteration = 0
        self.xdim = xdim
        self.ydim = ydim
        self.species = {}
        self.specieCtr = 0
        self.ybest = np.zeros((0, 1))
        self.ybest_adjusted = np.zeros((0, 1))
        self.npop = npop
        self.output_activation = ydim*[0] if output_activation is None else output_activation
        self.nparallel = nparallel
        self.maxtimelevel = maxtimelevel
        self.genomes = None

    ### Initialize ###
    def initialize(self):
        self.currentIteration = 0
        self.specieCtr = 0
        self.genomes = [Genom.initializeRandomly(ninputs=self.xdim, noutputs=self.ydim, maxtimelevel=self.maxtimelevel, output_activation=self.output_activation) for _ in range(self.npop)]
        self.species = {0: {"genome_prototype": copy.copy(self.genomes[0]), "genomes": [], "fitness_adjusted": None, "best_genom": self.genomes[0],
                       "fitness":[] , "best_fitness": -1e+4, "best_fitness_gen": 0, "noffspring": 0, "reproduce":True}}

    ### here goes your code ###
    def run(self, genom):
        pass

    ### Run different genomes for different generations ###
    def iterate(self, generationMax, sigmat=1.0, keepratio=0.2, maxsurvive=10, pcrossover=0.75, paddNode=0.05, prmNode=0.05, paddCon=0.07, prmCon=0.05, pmutW=0.8):

        ### Start the evolution ###
        for generation in range(generationMax):

            ### Clean species dict ###
            for specieID, specie in self.species.items():
                specie["genomes"], specie["fitness"] = [], []

            ### Run genomes ###
            print("##### Running Generation {} #####".format(generation))
            p = multiprocessing.Pool(processes=self.nparallel)
            fitness = p.map(self.run, [genom for genom in self.genomes])
            fitness = np.asarray(fitness).reshape(-1)

            ### Calculate compability distance ###
            for f,genom in zip(fitness, self.genomes):
                matched = False
                for ngenomes, specie in self.species.items():
                    if Genom.compabilityMeasure(genom, specie["genome_prototype"]) <= sigmat:
                        specie["genomes"].append(genom)
                        specie["fitness"].append(f)
                        matched = True
                        break
                if not matched:
                    self.specieCtr+=1
                    self.species[self.specieCtr] = {"genome_prototype": copy.copy(genom), "genomes": [genom], "fitness_adjusted": None, "best_genom": genom,
                                               "fitness":[f], "best_fitness": f, "best_fitness_gen": generation, "noffspring": 0, "reproduce": True}

            ### Adjust by group fitness and kill a certain % ###
            for specieID, specie in self.species.items():
                ### Check if specie is populated ### 
                ngenomes = len(specie["genomes"])
                if ngenomes == 0:
                    specie["noffspring"] = 0
                    continue

                ### Sort by fitness ###
                f = np.asarray(specie["fitness"]).reshape(-1)
                idxs = f.argsort()[::-1]
                ikill = int(keepratio*ngenomes) if int(keepratio*ngenomes) > 1 else 1

                ### Sort by fitness ###
                f = f[idxs]
                fadj = f/ngenomes
                specie["genomes"] = [specie["genomes"][idx] for idx in idxs]

                ### kill half of the population ###
                specie["fitness"] = f[:ikill].tolist()
                specie["genomes"] = specie["genomes"][:ikill]
                specie["fitness_adjusted"] = fadj[:ikill].tolist()
                specie["noffspring"] = np.mean(fadj[:ikill]) if specie["reproduce"] else -1e+3
                specie["genome_prototype"] = copy.copy(specie["genomes"][0])

                ### Check if specie has gotten better ###
                if f[0] > specie["best_fitness"]:
                    ### Specie has gotten better so we reset the best_fitness_gen to current generation ###
                    specie["best_fitness"], specie["best_fitness_gen"] = f[0], generation
                    specie["best_genom"] = copy.copy(specie["genomes"][0])
                    specie["reproduce"] = True
                elif (generation-specie["best_fitness_gen"]) > maxsurvive:
                    ### No improvement since Nextinguish generation ###
                    specie["best_fitness_gen"], specie["reproduce"], specie["noffspring"] = generation, False, -1e+3

                ### Print summary ###
                print("-> Specie: {}, Pop: {}, FitMin: {:.3f}/{:.3f}, FitMean: {:.3f}/{:.3f}, FitMax: {:.3f}/{:.3f}, Reprod: {}".format(specieID, ngenomes, f.min(), fadj.min(), f.mean(), fadj.mean(), f.max(), fadj.max(), specie["reproduce"]))


            ### Offspring ###
            pmate = np.asarray([specie["noffspring"] for specieID, specie in self.species.items()])
            pmate = np.exp(pmate) / np.sum(np.exp(pmate))

            ### Elitists ###
            self.genomes = [specie["best_genom"]for specieID, specie in self.species.items() if len(specie["genomes"]) > 0 and specie["reproduce"]]

            ### Next generation ###
            while len(self.genomes)<self.npop:
                #print(list(species.keys()), noffsprings, )
                specieID = int(np.random.choice(list(self.species.keys()), 1, pmate.tolist()))
                specie = self.species[specieID]

                ### Sanity check ###
                if len(specie["genomes"]) == 0 or not specie["reproduce"]:
                    continue

                ### Crossover ###
                elif len(specie["genomes"])>5 and specie["reproduce"] and np.random.rand()< pcrossover:
                    idxs = np.random.choice(len(specie["genomes"]), 2, replace=False)

                    if specie["fitness"][idxs[0]] > specie["fitness"][idxs[1]]:
                        genom1, genom2 = specie["genomes"][idxs[0]], specie["genomes"][idxs[1]]
                    else:
                        genom2, genom1 = specie["genomes"][idxs[0]], specie["genomes"][idxs[1]]

                    genom = Genom.crossover(genom1, genom2, generation=generation)

                ### Just take any random design and mutate it ###
                else:
                    idxs = np.random.choice(len(specie["genomes"]), 1)
                    genom = specie["genomes"][idxs[0]]

                ### Mutations ###
                for _ in range(1):
                    if np.random.rand() < paddNode:
                        genom = Genom.mutate_add_node(genom, generation=generation)
                    if np.random.rand() < prmNode:
                        genom = Genom.mutate_remove_node(genom, generation=generation)
                    if np.random.rand() < paddCon:
                        genom = Genom.mutate_add_connection(genom, generation=generation)
                    if np.random.rand() <prmCon:
                        genom = Genom.mutate_remove_connection(genom, generation=generation)
                    if np.random.rand() < pmutW:
                        genom = Genom.mutate_weight(genom, generation=generation)
                    if np.random.rand() < 0.1*pmutW:
                        pass #genom = Genom.mutate_bias(genom, generation=generation)

                ### Finally append it ###
                self.genomes.append(genom)

### Simulation environment for neat ###
def bestfit(genom):

    x = np.linspace(0,8,20).reshape(-1,1)
    y = 0.5*(np.sin(x)+1)

    xnorm = 0.1*(x-4)
    yhat = genom.run(xnorm)
    rmse = np.mean((y-yhat)**2)

    return 10.0-rmse


# ### Simulation environment for neat ###
def pendulum(genom, timesteps=350, render=False, seeds=[42,1337,87]):
    env = gym.make("Pendulum-v0")
    ep_reward = 0
    ylb, yub = np.asarray([-2.0]), np.asarray([2.0])
    xlb, xub = np.asarray([-1,-1,-8]), np.asarray([ 1, 1, 8])
    
    #print("Running genom {}".format(genom))
    for _, seed in enumerate(seeds):
        env.seed(seed)
        s = env.reset()
        for t in range(timesteps):

            ### Run single genomes ###
            s_norm = -1+2*(np.asarray(s).reshape(1,-1)-xlb)/(xub-xlb)
            s_norm = np.hstack((s_norm,np.ones((1,1))))
            a_norm = genom.run(s_norm)
            a = yub*a_norm.reshape(-1)

            ### Run simulation environment ###
            s2, r, done, info = env.step(a)


            r = s2[0] if s2[0]<0.7 else 3*s2[0]

            ep_reward += r
            s = s2
            ### plotting and stopping ###

            if render:
                env.render()

    return 10+ep_reward/len(seeds)/timesteps


def cartPole(genom, timesteps=400, render=False, repeat=15):
    ep_reward = 0
    env = gym.make("CartPole-v0")
    for _ in range(repeat):
        s = env.reset()
        for t in range(timesteps):

            ### Run single genomes ###
            #logits = genom.run(s.reshape(1,-1)).reshape(-1)
            #probs = np.exp(logits) / np.sum(np.exp(logits))
            #a = np.random.choice([0, 1], 1, p=probs)[0]

            logits = genom.run(s.reshape(1,-1)).reshape(-1)
            probs = np.exp(logits) / np.sum(np.exp(logits))
            a = np.random.choice([0, 1], 1, p=probs)[0]
            ### Run simulation environment ###
            s2, r, done, info = env.step(a)
            ep_reward += r
            s = s2
            ### plotting and stopping ###
            if done:
                break
            if render:
                env.render()

    return ep_reward/repeat


def acrobot(genom, timesteps=400, render=False, repeat=15):
    ep_reward = 0
    env = gym.make("Acrobot-v1")

    for _ in range(repeat):
        s = env.reset()
        for t in range(timesteps):

            ### Run single genomes ###
            logits = genom.run(s.reshape(1,-1)).reshape(-1)
            probs = np.exp(logits) / np.sum(np.exp(logits))
            a = np.random.choice([0, 1, 2], 1, p=probs)[0]

            ### Run simulation environment ###
            s2, r, done, info = env.step(a)
            ep_reward += r
            s = s2
            ### plotting and stopping ###
            if done:
                break
            if render:
                env.render()

    return timesteps+ep_reward/repeat

#### TEST #######
if __name__ == "__main__":

    if False:
        ### Use gym as test environment ###
        # ### Simulation environment for neat ###
        
        ### NEAT ###
        neat = NEAT(xdim=4, ydim=2, npop=100, maxtimelevel=1, output_activation=[0,0])
        neat.initialize()
        neat.run = cartPole
        neat.iterate(15, sigmat=2.5, maxsurvive=5)

        for specieID, specie in neat.species.items():
            if len(specie["genomes"])>0:
                neat.run(specie["best_genom"], render=True)
                specie["genomes"][0].showGraph()


    elif True:
        ### Use gym as test environment ###
        # ### Simulation environment for neat ###
        

        ### NEAT ###
        neat = NEAT(xdim=4, ydim=1, npop=100, maxtimelevel=1, output_activation=[2])
        neat.initialize()

        neat.run = pendulum
        neat.iterate(20, sigmat=1.0, keepratio=0.5, maxsurvive=500, paddNode=0.09, prmNode=0.08, paddCon=0.10, prmCon=0.10, pmutW=0.8, pcrossover=0.75)

        for specieID, specie in neat.species.items():
            if len(specie["genomes"])>0:
                specie["best_genom"].showGraph()
                print(specie["best_genom"].structure)
                neat.run(specie["best_genom"], render=True)
                
    elif False:
        ### Use gym as test environment ###
        # ### Simulation environment for neat ###
        
        ### NEAT ###
        neat = NEAT(xdim=6, ydim=3, npop=20, maxtimelevel=1, output_activation=[0,0,0])
        neat.run = acrobot
        neat.initialize()
        neat.iterate(15, sigmat=3.0)

        for specieID, specie in neat.species.items():
            if len(specie["genomes"])>0:
                neat.run(specie["best_genom"], render=True)
                specie["best_genom"].showGraph()


    else:
        ### NEAT ###
        neat = NEAT(xdim=1, ydim=1, npop=100, maxtimelevel=1, output_activation=[0])
        neat.run = bestfit
        neat.initialize()
        neat.iterate(80, sigmat=4.0, keepratio=0.2, maxsurvive=150, paddNode=0.05, prmNode=0.05, paddCon=0.1, prmCon=0.1, pmutW=0.8)



        for specieID, specie in neat.species.items():
            x = np.linspace(0,8,20).reshape(20,1)
            y = 0.5*(np.sin(x)+1)
            if len(specie["genomes"])>0:
                xnorm = 0.1*(x-4)
                yhat = specie["best_genom"].run(xnorm)
                print(specie["best_genom"])
                specie["best_genom"].showGraph()
                plt.plot(x, yhat,'ro-')
                plt.plot(x, y,'bo-')
                plt.show()
                
            else:
                continue
