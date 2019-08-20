#!/usr/bin/python
###############################################################################
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os
import re
import sys

# =======================================================
# =======================================================
#====================================================
# =======================================================
# =======================================================
class neural_network(object):
    # =======================================================
    def __init__(self,npop,ninput,noutput,connectioncost=0.0,nodecost=0.0,weightcost=0.0,sigma_tres=1.5,maxhiddennnodes=0):
        self.wmax=16
        self.bnodes=[ninput,noutput]
        self.npop=npop
        self.speciectr=0
        self.bestnetworkdict={}
        self.nodedict={}
        self.condict={}
        self.nodedict["ctr"]=0
        self.condict["ctr"]=0
        self.sigma_tres = sigma_tres

        self.costfac=[nodecost,connectioncost,weightcost]

        self.plotnames=""
        if maxhiddennnodes == 0:
            self.maxnnodes=ninput+noutput+2
        else:
            self.maxnnodes=ninput+noutput+maxhiddennnodes
            
        self.limitctr=0
        self.fitnesslog_best=[]
        self.fitnesslog_ave=[]
        self.fitnesslog_worst=[]
        self.structure_nodes=[]
        self.structure_cons=[]
        os.system("rm -f data_neatdna.dat")
        os.system("rm -f specie_evolve_logfile.txt")
        os.system("rm -f specie_logfile.txt")
    # =======================================================
    # Specie Initialization
    # =======================================================
    def initialization(self,bias_prob=0.05,con_prob = 0.2):
        #--------------------------------
        input_nids=list(range(1,self.bnodes[0]+1))
        output_nids=list(range(900,self.bnodes[1]+900))
        
        self.nodedict["ctr"]=len(input_nids)
        
        nids=input_nids[:]
        nids.extend(output_nids[:])
        ntype = self.bnodes[0]*["i"]
        ntype.extend(self.bnodes[1]*["e"])
        nfct = self.bnodes[0]*["c"]
        nfct.extend(self.bnodes[1]*["s"])

        #--------------------------------
        dnas=self.npop*['']
        #--------------------------------
        for n in range(self.npop):
            dna={}
            dna["parents"]=["",""]
            dna["generation"]=0
            dna["name"]=self.specie_ctr()
            dna["node_id"]=nids
            dna["node_typ"]=ntype
            dna["node_fct"]=nfct
            dna["survived"]=0
            dna["crossover"]=True
            
            #--------------------------------
            # BIAS is given to some designs
            nbias=len(nids)*[0.0]
            if np.random.rand() <= bias_prob:
                nbias[np.random.randint(self.bnodes[0],self.bnodes[0]+self.bnodes[1])]=self.random_weight()
            dna["node_bias"]=nbias
            #--------------------------------
            for i in input_nids:
                dna["c"+str(i)]=[[],[]]
            #--------------------------------
            for e in output_nids:
                inids,cnws=[],[]
                for i in input_nids:
                    if np.random.rand() <= con_prob:
                        inids.append(i)
                        cnws.append(self.random_weight())

                dna["c"+str(e)]=[inids,cnws]
            #--------------------------------
            #--------------------------------
            dnas[n]=dna
        #--------------------------------
        # Add innovation numbers to basic DNA
        for k,dna in enumerate(dnas):
            dnas[k]=self.check_innovationnumbers(dna)
        return dnas

    # =======================================================
    # =======================================================
    # Invention manager
    # =======================================================
    # =======================================================
    def invention_manager(self,n1,n2,inodict={}):
        # Keeps track of innovations and assignes protected numbers to it!
        name = str(n1)+"->"+str(n2)
        try:
            ino=inodict[name]
        except:
            ino=inodict["ctr"]+1
            inodict["ctr"]=ino
            inodict[name]=ino
            inodict[ino]=[n1,n2]
            print("\t\tNew innovation number ("+str(ino)+") assigned to "+name)
        return ino
        
    # =======================================================
    def check_innovationnumbers(self,dna):
        # This function goes through the DNA and looks for new 
        # innovations. New innovations are protected by the manager then.
        # The DNA then gets its list of innovation numbers needed for crossover.
        inos=[]
        for n in range(self.bnodes[0],len(dna["node_id"])):
            rnid = dna["node_id"][n]
            inids,inws = dna["c"+str(rnid)][0],dna["c"+str(rnid)][1]

            for inid, w in zip(inids, inws):
                inos.append(self.invention_manager(inid,rnid,inodict=self.condict))
        dna["ino"] = inos
        return dna
    
    # =======================================================
    # =======================================================
    # Feed Forward Network Evaluation
    # =======================================================
    # =======================================================
    def forward_propagation(self,dna,X,debug=False):
        # This function conductes forwards propagation for the given dna and calculates
        # the node states.
        #------------------------------------------------
        # Initialize 
        #------------------------------------------------
        if debug:
            f1 = open("specie_logfile.txt","a")
            f1.write(""+50*"-"+"\n")
            f1.write("Forward propagation DNA: "+str(dna["name"])+"\n")
            f1.write(""+50*"-"+"\n")
        #------------------------------------------------
        # Get prediction data dimensions
        #------------------------------------------------
        nsmpls,ninput_smpls=X.shape[0],X.shape[1]
        #------------------------------------------------
        # DNA sanity checks
        #------------------------------------------------    
        assert len(dna["node_id"]) == len(dna["node_typ"]), "Exiting: Node type error!"
        assert len(dna["node_id"]) == len(dna["node_fct"]), "Exiting: Node fct error!"
        assert len(dna["node_id"]) == len(dna["node_bias"]), "Exiting: Node bias error!"
        
        input_index=[pos for pos, char in enumerate(dna["node_typ"]) if char == "i"]
        output_index=[pos for pos, char in enumerate(dna["node_typ"]) if char == "e"] 
        
        assert len(input_index) == ninput_smpls, "Exiting: Input data error!"
        assert ninput_smpls == self.bnodes[0], "Exiting: Input data error!"
        #------------------------------------------------
        # Initialize node states
        #------------------------------------------------
        nodes_state = np.zeros((nsmpls,len(dna["node_id"])))
        #------------------------------------------------
        # Assign input to input nodes
        #-----------------------------------------------
        nodes_state[:,input_index]=X
        #------------------------------------------------
        # Forward Prediction
        #------------------------------------------------
        for n in range(self.bnodes[0],len(dna["node_id"])):
            rnid = dna["node_id"][n]
            
            if debug:
                f1.write("\t"+50*"-"+"\n")
                f1.write("\tPrediction for node: "+str(rnid)+"\n")
            
            inids,inws = dna["c"+str(rnid)][0],dna["c"+str(rnid)][1]

            if inids == [] and debug:
                f1.write("\t\t\tWarning: Node "+str(rnid)+" not used!"+"\n")
                
            # Add bias to node state
            nodes_state[:,n]+=dna["node_bias"][n]
            #------------------------------------------------
            for inid, w in zip(inids, inws):
                
                # Feedforward Check
                if dna["node_id"].index(inid) >= dna["node_id"].index(rnid) and debug:
                    print("Serious DNA Error: System is not feedforward for specie "+str(dna["name"])+"! Aborting!")
                    sys.exit()            
                try:
                    if debug:
                        f1.write("\t\tGet connection: "+str(inid)+" -> "+str(rnid)+"\t\t("+str(w)+")"+"\n")
                    inid_index = dna["node_id"].index(inid)
                    nodes_state[:,n]+= w *nodes_state[:,inid_index]
                except:
                    # If there is no connection to receiving node
                    if debug:
                        f1.write("\t\t\tWarning: incoming_node "+str(inid)+" not used!"+"\n")
            #------------------------------------------------
            # Apply defined activation function
            #------------------------------------------------        
            if dna["node_fct"][n] == "s":
                nodes_state[:,n] = self.sigmoid(nodes_state[:,n])
            elif dna["node_fct"][n] == "c":
                nodes_state[:,n] = self.constant(nodes_state[:,n],fac=0.1)
                #pass
            else:
                "ERROR ACTIVATION FUNCTION NOT FOUND!"
                sys.exit()
            if debug:
                f1.write("\t\tActivation function ("+str(dna["node_fct"][n])+") applied!"+"\n")
        #------------------------------------------------
        # Return only the node states of the output nodes
        #------------------------------------------------
        if debug:
            f1.close()
        return nodes_state[:,output_index]
        #-----------------------------------------------

    #=======================================================
    #=======================================================
    # Mutation add: node operator
    #=======================================================
    #=======================================================
    def dna_add_node(self,dna,debug=False):
        if len(dna["node_id"]) < self.maxnnodes:
            #------------------------------------------------        
            #Mutation operator, replaces a connection with a node + two connections!
            #------------------------------------------------        
            dna_mut={}
            dna_mut["survived"]=0
            dna_mut["crossover"]=True
            dna_mut["name"]=self.specie_ctr()
            dna_mut["generation"]=dna["generation"]
            dna_mut["parents"]=[dna["name"],"node_mutated"]
            
            #------------------------------------------------
            if debug:
                f1 = open("specie_evolve_logfile.txt","a")
                f1.write(""+50*"-"+"\n")
                f1.write("Mutation add node DNA: "+str(dna_mut["name"])+"\n")
                f1.write(""+50*"-"+"\n")
            #------------------------------------------------
            nids_org = dna["node_id"]
            ino_org = dna["ino"]
            
            hidden_index_org=[pos for pos, char in enumerate(dna["node_typ"]) if char == "h"]

            nhidden = len(hidden_index_org)
            nnodes_org=len(nids_org)

            if hidden_index_org == []:
                nid_r_index=self.bnodes[0]
            else:
                nid_r_index=hidden_index_org[np.random.randint(nhidden)]
                
            nid_l_index=nid_r_index-1

            nid_mut = self.invention_manager(nids_org[nid_l_index],nids_org[nid_r_index],inodict=self.nodedict)
            if debug:
                f1.write("\tAdd new node : "+str(nid_mut)+"\n")

            for key,val in zip(["node_id","node_fct","node_typ","node_bias"],[nid_mut,"s","h",0.0]):
                dna_mut[key]=dna[key][:nid_r_index]
                dna_mut[key].append(val)
                dna_mut[key].extend(dna[key][nid_r_index:])
            #--------------------------------------------------------
            for nid in nids_org:
                dna_mut["c"+str(nid)]=dna["c"+str(nid)][:]
            #--------------------------------------------------------
            # Link new nodes with connections
            dna_mut["c"+str(nid_mut)]=[[],[]]
            
            nid_mut_index = dna_mut["node_id"].index(nid_mut)
                
            inid = dna_mut["node_id"][np.random.randint(nid_mut_index)]
            rnid = dna_mut["node_id"][np.random.randint(nid_mut_index+1,nnodes_org+1)]
                

            inids,incws=dna_mut["c"+str(rnid)][0][:], dna_mut["c"+str(rnid)][1][:]
            
            
            if inid in inids:
                index=inids.index(inid)
                dna_mut["c"+str(nid_mut)]=[[inid],[2.0]] #2
                inids.append(nid_mut)
                incws.append(incws[index])

                inids.pop(index)
                incws.pop(index)
                if debug:
                    f1.write("\tRemove connection: "+str(inid)+"->"+str(nid_mut)+"\n")

            else:
                dna_mut["c"+str(nid_mut)]=[[inid],[self.random_weight()]]
                inids.append(nid_mut)
                incws.append(self.random_weight())
                
            dna_mut["c"+str(rnid)]=[inids,incws]
            if debug:
                f1.write("\tAdd connection: "+str(inid)+"->"+str(nid_mut)+"\n")
                f1.write("\tAdd connection: "+str(nid_mut)+"->"+str(rnid)+"\n")
            #--------------------------------------------------------
            # Check innovation numbers
            dna_mut=self.check_innovationnumbers(dna_mut)
            #--------------------------------------------------------
            if debug:
                f1.close()
            return dna_mut
            #--------------------------------------------------------
        else:
            return dna
    #=======================================================
    #=======================================================
     # Mutation: add connection operator
    #=======================================================
    #=======================================================
    def dna_add_connection(self,dna,debug=False,addconprob_tres =0.8):

        nids = dna["node_id"][:]
        dna_mut={}
        dna_mut["generation"]=dna["generation"]
        dna_mut["parents"]=[dna["name"],"con_mutated"]
        dna_mut["name"]=self.specie_ctr()
        dna_mut["survived"]=0
        dna_mut["crossover"]=True
        
        for val in ["node_id","node_fct","node_typ","node_bias"]:
            dna_mut[val]=dna[val][:]

        for nid in nids:
            dna_mut["c"+str(nid)]=dna["c"+str(nid)][:]
        #------------------------------------------------
        if debug:
            f1 = open("specie_evolve_logfile.txt","a")
            f1.write(""+50*"-"+"\n")
            f1.write("Mutation add connection: "+str(dna["name"])+"\n")
            f1.write(""+50*"-"+"\n")
        #------------------------------------------------
        addconprob = np.random.rand()

        for contry in range(20):
            
            rnid_index = np.random.randint(self.bnodes[0],len(nids))
            rnid = nids[rnid_index]
            inid_index = np.random.randint(rnid_index)
            inid = nids[inid_index]
        
            inids=dna["c"+str(rnid)][0][:]
            incws=dna["c"+str(rnid)][1][:]
            incw = (self.random_weight())

            if not inid in inids and not inid in range(900,self.bnodes[1]+900) and addconprob<addconprob_tres:
                inids.append(inid)
                incws.append(incw)
                dna_mut["c"+str(rnid)]=[inids,incws]
                if debug:
                    f1.write("\tAdd connection: "+str(inid)+"->"+str(rnid)+"\n")
                break
                
            elif inid in inids and len(inids)>1 and addconprob>addconprob_tres:
                index=inids.index(inid)
                inids.pop(index)
                incws.pop(index)
                dna_mut["c"+str(rnid)]=[inids,incws]
                if debug:
                    f1.write("\tRemove connection: "+str(inid)+"->"+str(rnid)+"\n")
                break
        #--------------------------------------------------------
        # Check innovation numbers
        dna_mut=self.check_innovationnumbers(dna_mut)
        #--------------------------------------------------------
        if debug:
            f1.close()
        return dna_mut
    #=======================================================
    #=======================================================
     # Mutation: mutate weight
    #=======================================================
    #=======================================================
    def dna_mutate_weight(self,dna,debug=False):
    
        nids = dna["node_id"][:]
        dna_mut={}
        dna_mut["generation"]=dna["generation"]
        dna_mut["parents"]=[dna["name"],"w_mutated"]
        dna_mut["name"]=self.specie_ctr()
        dna_mut["survived"]=0
        dna_mut["crossover"]=True
        #------------------------------------------------
        if debug:
            f1 = open("specie_evolve_logfile.txt","a")
            f1.write(""+50*"-"+"\n")
            f1.write("Mutation weight DNA: "+str(dna_mut["name"])+"\n")
            f1.write(""+50*"-"+"\n")
        #------------------------------------------------
        for val in ["node_id","node_fct","node_typ","node_bias"]:
            dna_mut[val]=dna[val][:]
            
        for nid in nids:
            dna_mut["c"+str(nid)]=dna["c"+str(nid)][:]
            
        for contry in range(10):
            rnid_index = np.random.randint(self.bnodes[0],len(nids))
            rnid = nids[rnid_index]
           
            inids=dna_mut["c"+str(rnid)][0][:]
            incws=dna_mut["c"+str(rnid)][1][:]
            
            if not inids == []:
                inid_index=np.random.randint(len(inids))
                
                incw0=incws[inid_index]
                
                if np.random.rand() > 0.9:
                    incw = incw0+(-1.0 + 2*np.random.rand())
                    if incw >self.wmax:
                        incw=self.wmax
                    elif incw < -self.wmax:
                        incw=-self.wmax
                else:
                    incw = (self.random_weight())
                
                incws[inid_index]=incw
                dna_mut["c"+str(rnid)]=[inids,incws]
                break
        #--------------------------------------------------------
        # Check innovation numbers
        dna_mut=self.check_innovationnumbers(dna_mut)
        #--------------------------------------------------------
        if debug:
            f1.close()
        return dna_mut
    #=======================================================
    #=======================================================
     # Mutation: mutate bias
    #=======================================================
    #=======================================================
    def dna_mutate_bias(self,dna,debug=False):
    
        nids = dna["node_id"][:]
        dna_mut={}
        dna_mut["generation"]=dna["generation"]
        dna_mut["parents"]=[dna["name"],"bias_mutated"]
        dna_mut["name"]=self.specie_ctr()
        dna_mut["survived"]=0
        dna_mut["crossover"]=True
        #------------------------------------------------
        if debug:
            f1 = open("specie_evolve_logfile.txt","a")
            f1.write(""+50*"-"+"\n")
            f1.write("Mutation bias DNA: "+str(dna_mut["name"])+"\n")
            f1.write(""+50*"-"+"\n")
        #------------------------------------------------
        for val in ["node_id","node_fct","node_typ","node_bias"]:
            dna_mut[val]=dna[val][:]
            
        for nid in nids:
            dna_mut["c"+str(nid)]=dna["c"+str(nid)][:]
        #------------------------------------------------
        rnid_index = np.random.randint(self.bnodes[0],len(nids))
        rnid = dna_mut["node_id"][rnid_index]
        
        if dna_mut["node_bias"][rnid_index] == 0 and len(dna_mut["c"+str(rnid)][0])>1:
            dna_mut["node_bias"][rnid_index]=self.random_weight()
            
        elif not dna_mut["node_bias"][rnid_index] == 0 and np.random.rand()>0.5 and len(dna_mut["c"+str(rnid)][0])>1:
            dna_mut["node_bias"][rnid_index]=dna_mut["node_bias"][rnid_index] +(-1.0 +2*np.random.rand())
        else: 
            dna_mut["node_bias"][rnid_index]=0
        #--------------------------------------------------------
        # Check innovation numbers
        dna_mut=self.check_innovationnumbers(dna_mut)
        #--------------------------------------------------------
        if debug:
            f1.close()
        return dna_mut
    #=======================================================
    #=======================================================
    # Crossover operator
    #=======================================================
    #=======================================================
    def dna_crossover(self,dna1,dna2,debug=False):

        #------------------------------------------------        
        # Crossover Operator: dna1 one is expected to be the fitter dna
        #------------------------------------------------        
        dna_child={}
        dna_child["name"]=self.specie_ctr()
        dna_child["generation"]=max([dna1["generation"],dna2["generation"]])+1
        dna_child["parents"]=[dna1["name"],dna2["name"]]
        dna_child["survived"]=0
        dna_child["crossover"]=True
        #------------------------------------------------        
        if debug:
            f1 = open("specie_evolve_logfile.txt","a")
            f1.write(""+50*"-"+"\n")
            f1.write("Crossover: Mate  "+str(dna1["name"])+"x"+str(dna2["name"])+" -> "+str(dna_child["name"])+"\n")
            f1.write(""+50*"-"+"\n")
        #------------------------------------------------
        # Find common and uncommon nodes
        #------------------------------------------------            
        dna_child["node_id"] = dna1["node_id"][:]
        dna_child["ino"] = dna1["ino"][:]
        
        cinos = list(set(dna1["ino"][:]).intersection(dna2["ino"][:]))
        ucinos = [x for x in dna_child["ino"] if x not in cinos]
        
        # Crossover points
        cois=np.zeros(len(cinos))
        
        if not len(cinos)==0:
            coi=int(len(cinos)/2) #+ np.random.randint(-len(cinos)/2,len(cinos)/2)
        else:
            coi=0
            

        cois[:coi],cois[coi:]=0,1
        
        dna_child["node_fct"]=len(dna_child["node_id"])*['']
        dna_child["node_bias"]=len(dna_child["node_id"])*['']
        dna_child["node_typ"]=len(dna_child["node_id"])*['']
        #------------------------------------------------        
        # Get Bias, fct and typ from parents
        #------------------------------------------------            
        for n,nid in enumerate(dna_child["node_id"]):
            try:
                index=dna1["node_id"].index(nid)
                for key in ["node_typ","node_fct","node_bias"]:
                    dna_child[key][n]=dna1[key][index]
            except:
                index=dna2["node_id"].index(nid)
                for key in ["node_typ","node_fct","node_bias"]:
                    dna_child[key][n]=dna2[key][index]
        #------------------------------------------------            
        # Add connections from inovations to dna
        #------------------------------------------------    
        # Initialize connections
        for nid in dna_child["node_id"]:
            dna_child["c"+str(nid)]=[[],[]]
            
        #------------------------------------------------   
        for n,cino in enumerate(cinos):
            
            [n1,n2]=self.condict[cino]
            assert dna_child["node_id"].index(n2) > dna_child["node_id"].index(n1), "System not feed forward! Abborting!"
            
            icnl,icnw=dna_child["c"+str(n2)][0],dna_child["c"+str(n2)][1]
            icnl.append(n1)
            
            if cois[n] == 0:
                picnl_i=dna1["c"+str(n2)][0].index(n1)
                icnw.append(dna1["c"+str(n2)][1][picnl_i])
                if debug:
                    f1.write("\t("+str(n1)+"->"+str(n2)+"): Taking weights from parent 1"+"\n")
            elif cois[n] == 1:
                picnl_i=dna2["c"+str(n2)][0].index(n1)
                icnw.append(dna2["c"+str(n2)][1][picnl_i])
                if debug:
                    f1.write("\t("+str(n1)+"->"+str(n2)+"): Taking weights from parent 2"+"\n")
                    
            dna_child["c"+str(n2)]=[icnl,icnw]
        #------------------------------------------------   
        for n,ucino in enumerate(ucinos):
            
            [n1,n2]=self.condict[ucino]
            assert dna_child["node_id"].index(n2) > dna_child["node_id"].index(n1), "System not feed forward! Abborting!"
            
            icnl,icnw=dna_child["c"+str(n2)][0],dna_child["c"+str(n2)][1]
            icnl.append(n1)
            
            picnl_i=dna1["c"+str(n2)][0].index(n1)
            icnw.append(dna1["c"+str(n2)][1][picnl_i])
            if debug:
                f1.write("\t("+str(n1)+"->"+str(n2)+"): Taking weights from parent 1"+"\n")
            
            dna_child["c"+str(n2)]=[icnl,icnw]
        #------------------------------------------------
        
        
        #------------------------------------------------
        if debug:
            f1.close()
        return dna_child
        
    #=======================================================
    # ======================================================
    # GA evolution stuff
    # ======================================================
    #=======================================================
    def initialize_optimization(self,X,Y):
        self.fitness_best=1e+26
        self.fitness_worst=0
        self.fitness_ave=0
        self.X=X
        self.Y=Y

    # =======================================================
    def fitnesslog(self):
        self.fitnesslog_best.append(self.fitness_best)
        self.fitnesslog_worst.append(self.fitness_worst)
        self.fitnesslog_ave.append(self.fitness_ave)

        self.structure_nodes.append(len(self.bestmember["node_id"]))
        self.structure_cons.append(len(self.bestmember["ino"]))
        
        fig=plt.figure()

        ax1 = fig.add_subplot(1, 1, 1)
        ax2 = ax1.twinx()
        
        ax1.semilogy(self.fitnesslog_ave,'k-')
        ax1.semilogy(self.fitnesslog_worst,'r-')
        ax1.semilogy(self.fitnesslog_best,'b-', lw=2)
        

        ax2.plot(self.structure_nodes,'g-',lw=2)
        ax2.plot(self.structure_cons,'g--',lw=2)
        
        plt.xlabel("Iterations")
        #plt.ylabel("Fitness")
        ax1.set_ylabel("Fitness", color='b')
        ax2.set_ylabel("Structure", color='g')
        
        plt.grid(True)
        
        plt.savefig("plot_convergence_iter.png", dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format="png",
        transparent=False, bbox_inches='tight', pad_inches=0.1,
        frameon=None)
        plt.close()
    # =======================================================
    def plotbestnetworks(self):
        os.system("rm -f plot_best_network_specie_*")
        for specie in  self.bestnetworkdict.keys():
            self.shownetwork(self.bestnetworkdict[specie],plotname="./plot_best_network_specie_"+str(specie)+".png",add2mov=False)

    # =======================================================
    def write_dnadatabase(self,pop):
        keys=sorted(pop[0].keys())
        if not os.path.isfile("data_neatdna.dat"):
            line="# "
            for key in keys:
                line=line+"\t"+key
        else:
            line=50*"-"        
        f1=open("data_neatdna.dat","a")
        f1.write(line+"\n")
        
        for dna in pop:
            line=""
            for key in keys:
                line=line+"\t"+str(dna[key])
            f1.write(line+"\n")
        f1.close()
    # =======================================================
    def fun_evolve(self,pop,iteration=0,mutate_con=0.04,mutate_node=0.02,mutate_w=0.11,mutate_bias=0.04,speciate=True):
        #-------------------------------------
        # Get new fitness
        #-------------------------------------
        fitness_best_old = self.fitness_best

        fitness=np.asarray([self.prediction_error(pop[x],self.X,self.Y) for x in range(self.npop)])

        if speciate:
            sh=self.sharing_matrix(pop) # Sharing matrix
        else:
            sh=np.ones((self.npop,self.npop))
                
        fitness_adjusted=fitness[:]*np.sum(sh,axis=0) # Shared fitness

        ranking=np.argsort(fitness)
        ranking_adjusted=np.argsort(fitness_adjusted)
        #-------------------------------------
        self.fitness_ave= np.mean(fitness)
        self.fitness_best = fitness[ranking][0]
        self.fitness_worst = fitness[ranking][-1]
        
        self.bestnetworkdict={}
        self.bestmember=pop[ranking[0]]
        #-------------------------------------
        # Write to dna database
        # self.write_dnadatabase(pop)
        #-------------------------------------
        # Rank population and assign probabilities
        #-------------------------------------
        proba_ranked=self.get_probability_list(self.npop)
        rankedpop=[pop[x] for x in ranking_adjusted]
        #-------------------------------------
        # Increase survival counter
        #-------------------------------------
        for dna in pop:
            dna["survived"]+=1
            #if dna["survived"] > 15:
                #print("Specie "+str(dna["name"])+" is not allowed to crossover anymore!"
                #dna["crossover"]=False
        #-------------------------------------
        # Elitists - Save best dna with different number of nodes
        #-------------------------------------
        pop_next = [pop[ranking[0]]]
        already_seen_index = [ranking[0]]
        self.bestnetworkdict[0]=pop[ranking[0]]
        #print("Adding specie\t "+str(pop[ranking[0]]["name"])+" (fitness rank "+str(0)+")\tas "+str(0)+" specie! Nnodes: "+str(len(pop[ranking[0]]["node_id"]))+" Survived: "+str(pop[ranking[0]]["survived"])
        specie_ctr=1

        for n in range(1,self.npop):
            index_current=ranking[n]
            specie_occurence=np.sum(sh[index_current,:])
        
            if np.all(sh[index_current,already_seen_index] == 0) and specie_occurence >= 5:
                pop_next.append(pop[index_current])
                self.bestnetworkdict[specie_ctr]=pop[index_current]
                #print("Adding specie\t "+str(pop[index_current]["name"])+" (fitness rank "+str(n)+")\tas "+str(specie_ctr)+" specie! Nnodes: "+str(len(pop[index_current]["node_id"]))+" Survived: "+str(pop[index_current]["survived"])
                already_seen_index.append(index_current)
                specie_ctr+=1
            
            # If get stuck, not allowed to mate!
#            if pop[index_current]["survived"] > 20:
#                index_mateforbid=np.where(sh[index_current,:]==1)[0]

#                for m in range(index_mateforbid.shape[0]):
#                    print("Forbid design "+str(index_mateforbid[m])+"to mate"
#                    pop[index_mateforbid[m]]["crossover"]=False

            if specie_ctr == 10:
                break
        #-------------------------------------
        # Plot best networks
        #-------------------------------------
        if np.around(self.fitness_best,decimals=4) < np.around(fitness_best_old,decimals=4):
            self.limitctr = 0
            self.plotbestnetworks()
            self.shownetwork(self.bestmember,interactive=False)
        else:
            self.limitctr += 1
            
#        if self.limitctr == 20:
#            self.maxnnodes+=1
#            print("Increasing maximum node limit to "+str(self.maxnnodes)
        #-------------------------------------
        # Crossover
        #-------------------------------------
        desired_length = self.npop - len(pop_next)
        children = []
        
        while len(children) < desired_length:
            [parent_index1,parent_index2]=self.roulette_wheel_pop(proba_ranked,2)
            if parent_index1 > parent_index2:
                male=rankedpop[parent_index1]
                female=rankedpop[parent_index2]
            else:
                male=rankedpop[parent_index1]
                female=rankedpop[parent_index2]
                
            if male["crossover"] and female["crossover"] and sh[parent_index1,parent_index2] == 1 :

                children.append(self.dna_crossover(male,female))
                    
        pop_next.extend(children[:desired_length])
        #-------------------------------------
        # Mutate some members
        #-------------------------------------
        for n in range(1,len(pop_next)):
        
            if mutate_con > np.random.rand():
                pop_next[n]=self.dna_add_connection(pop_next[n])
                
            if mutate_node > np.random.rand():
                pop_next[n]=self.dna_add_node(pop_next[n])
                
            if mutate_bias > np.random.rand():
                pop_next[n]=self.dna_mutate_bias(pop_next[n])
                
            if mutate_w > np.random.rand():
                pop_next[n]=self.dna_mutate_weight(pop_next[n])

        #-------------------------------------
        return pop_next
    # =======================================================
    # =======================================================
    def fun_print_ave_fitness(self,n):
        ave = '%.3e' % self.fitness_ave
        best = '%.3e' % self.fitness_best
        worst = '%.3e' % self.fitness_worst
        print(str(n)+"\t"+str(ave)+"\t"+str(best)+"\t"+str(worst))
        return [self.fitness_ave,self.fitness_best,self.npop]
    # =======================================================
    def return_best_member(self):
        return self.bestmember
    # =======================================================
    # =======================================================
    # Compability Function
    # =======================================================
    # =======================================================
    def check_compability(self,dna1,dna2):
        #-------------------------------------
        c1,c2,c3=1.0,1.0,0.3
        #-------------------------------------
        averaged_conweight=2*['']
        for d,dna in enumerate([dna1,dna2]):
            sumconweights=0.0
            ncons=0
            
            for n,rnid in enumerate(dna["node_id"]):
                inids,inws = dna["c"+str(rnid)][0],dna["c"+str(rnid)][1]
                sumconweights+=sum(map(abs, inws))
                ncons+=len(inids)
            
            sumbias=sum(map(abs, dna["node_bias"]))
            
            if ncons == 0:
                averaged_conweight[d]=(sumconweights)+sumbias/len(dna["node_id"])
            else:
                averaged_conweight[d]=(sumconweights)/(ncons)+sumbias/len(dna["node_id"])

        W = abs(averaged_conweight[0]-averaged_conweight[1])

        #-------------------------------------
        ino1,ino2 = sorted(dna1["ino"][:]),sorted(dna2["ino"][:])
        N = max([len(ino1),len(ino2)])
        cino = [x for x in ino1 if x in ino2 ]       # common elements
        ucinos = [[x for x in ino1 if not x in ino2 ],[x for x in ino2 if not x in ino1 ] ] # uncommon elements in 1

        ndis,nex=0,0
        
        for n,ino in enumerate([ino1,ino2]):
            if cino == []:
                nex=len(ino1)+len(ino2)
            else:
                cino_index_0 = ino.index(cino[0])
                cino_index_1 = ino.index(cino[-1])
            
                for ucino in ucinos[n]:
                    ucino_index = ino.index(ucino)

                    if ucino_index > cino_index_0 and cino_index_0 < cino_index_1:
                        ndis+=1
                    else:
                        nex+=1     

        #-------------------------------------
        sigma = c1*ndis/(N+1e-20)+c2*nex/(N+1e-20) + c3*W
        #-------------------------------------
        if self.sigma_tres > sigma:
            return 1
        else:
            return 0
            
    # =======================================================
    def sharing_matrix(self,pop):

        sh=np.zeros((len(pop),len(pop)))

        for i,dna_i in enumerate(pop):
            sh[i,i]=1
            for j,dna_j in enumerate(pop):
                if i<j:
                    sh[i,j]=self.check_compability(dna_i,dna_j)
                    sh[j,i]=sh[i,j]

        return sh
    # =======================================================
    # =======================================================
    # Roulette Wheel
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
#    # =======================================================
#    # =======================================================
#    # GET PARETO RANKING
#    # =======================================================
#    # =======================================================
#    def calc_pareto_rank(self,Y):
#        #------------------------------------------------
#        def dominates_check(row, rowCandidate):
#            return all(r <= rc for r, rc in zip(row, rowCandidate))

#        def cull(pts,pts_index, dominates):
#            dominated = []
#            dominated_index = []
#            cleared = []
#            cleared_index = []
#            remaining = pts
#            remaining_index = pts_index
#            
#            while remaining:
#                candidate = remaining[0]
#                candidate_index = remaining_index[0]
#                new_remaining = []
#                new_remaining_index = []
#                
#                for other,other_index in zip(remaining[1:],remaining_index[1:]):
#                    [new_remaining, dominated][dominates(candidate, other)].append(other)
#                    [new_remaining_index, dominated_index][dominates(candidate, other)].append(other_index)
#                    
#                if not any(dominates(other, candidate) for other in new_remaining):
#                    cleared.append(candidate)
#                    cleared_index.append(candidate_index)
#                else:
#                    dominated.append(candidate)
#                    dominated_index.append(candidate_index)
#                    
#                remaining = new_remaining
#                remaining_index = new_remaining_index
#            return cleared_index, dominated,dominated_index
#        #------------------------------------------------
#        dominated=Y[:].tolist()
#        dominated_index=range(Y.shape[0])
#        rank=0
#        paretoranks=Y.shape[0]*[""]
#        
#        while not dominated == []:
#            [rankedpts, dominated,dominated_index]= cull(dominated,dominated_index, dominates_check)
#            for rankedpt in rankedpts:
#                paretoranks[rankedpt]=rank
#            rank+=1
#        #------------------------------------------------
#        return paretoranks
    # =======================================================
    # =======================================================
    # Plot the network
    # =======================================================
    # =======================================================
    def shownetwork(self,dna,interactive=False,plotname="",add2mov=True):
        # Plot the network
        #--------------------------------------------------
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        input_index=[pos for pos, char in enumerate(dna["node_typ"]) if char == "i"]
        output_index=[pos for pos, char in enumerate(dna["node_typ"]) if char == "e"]
        #--------------------------------------------------
        layerno=len(dna["node_id"])*[""]
        y=len(dna["node_id"])*[""]

        for n in input_index:
            layerno[n]=0
            y[n]=len(input_index)/2+n
        #--------------------------------------------------
        for n in range(len(input_index),len(dna["node_id"])):
            rnid = dna["node_id"][n]
            inids= dna["c"+str(rnid)][0]
            icnws= dna["c"+str(rnid)][1]
                
            maxpreviouslayerctr=0
            for inid, w in zip(inids, icnws):
                index=dna["node_id"].index(inid)
                if maxpreviouslayerctr<layerno[index]:
                    maxpreviouslayerctr=layerno[index]
            layerno[n]=maxpreviouslayerctr+1
        #--------------------------------------------------
        maxlayerno = max(layerno)
        for n in output_index:
            layerno[n]=maxlayerno
        #--------------------------------------------------
        for layer in range(maxlayerno+1):
            layer_indices=[pos for pos, char in enumerate(layerno) if char == layer]
            
            for n,layer_index in enumerate(layer_indices):
                if layer == maxlayerno or layer == 0:
                    y[layer_index] = (-len(layer_indices)*0.5+n)
                elif layer % 2 == 0:
                    y[layer_index] = (-len(layer_indices)*0.5+n)+0.1
                else:
                    y[layer_index] = (-len(layer_indices)*0.5+n)-0.1
        #--------------------------------------------------
        # Plots connections
        for n in range(len(dna["node_id"])):
            rnid = dna["node_id"][n]
            inids= dna["c"+str(rnid)][0]
            icnws= dna["c"+str(rnid)][1]
            bias=  dna["node_bias"][n]
            #--------------------------------------------------
            if abs(bias) > 0.0:
                if bias>=0:
                    plt.plot([layerno[n]-0.25,layerno[n]],[y[n]-0.25,y[n]],'r-',lw=3*(abs(bias)+1)/(self.wmax+1))
                else:
                    plt.plot([layerno[n]-0.25,layerno[n]],[y[n]-0.25,y[n]],'b-',lw=3*(abs(bias)+1)/(self.wmax+1))
            #--------------------------------------------------        
            for inid, w in zip(inids, icnws):
                index=dna["node_id"].index(inid)
                
                #plt.text(0.5*(layerno[n]+layerno[index]),0.5*(y[n]+y[index]),self.invention_manager(inid,rnid,inodict=self.condict), fontsize=8)
                
                if w>=0.0:
                    plt.plot([layerno[n],layerno[index]],[y[n],y[index]],'r-',lw=3*(abs(w)+1)/(self.wmax+1))
                else:
                    plt.plot([layerno[n],layerno[index]],[y[n],y[index]],'b-',lw=3*(abs(w)+1)/(self.wmax+1))
        #--------------------------------------------------
        # Plots nodes and bias
        for n in range(len(dna["node_id"])):
            plt.plot(layerno[n],y[n],'ko',markersize=10)
            #plt.text(layerno[n],y[n],dna["node_id"][n],color='w',ha='center', va='center')
            if abs(dna["node_bias"][n]) > 0.0:
                plt.plot(layerno[n]-0.25,y[n]-0.25,'k^',markersize=8)
        #--------------------------------------------------
        
        #--------------------------------------------------
        plt.axis([-0.1,maxlayerno+0.1,min(y)-0.1,max(y)+0.1])
        
        string="Specie: "+str(dna["name"])+"\n"
        string=string+"Number of nodes: "+str(len(dna["node_id"]))+"\n"
        string=string+"Number of connections: "+str(len(dna["ino"]))+"\n"
        string=string+"Generation:  "+str(dna["generation"])+"\n"
        string=string+"Parents:   "+str(dna["parents"][0])+" x "+str(dna["parents"][1])+"\n"
        #string=string+"Survived:   "+str(dna["survived"])+"\n"
        
        try:
            for n,fitness in enumerate(dna["fitness"]):
                string=string+"Fitness_"+str(n)+":  "+str(np.around(fitness,decimals=4))+"\n"
        except:
            pass
        #plt.annotate(string, (0.8,0), (0, 25), xycoords='axes fraction', textcoords='offset points', fontsize=8,ha='center', va='center', bbox=props)
        plt.axis('off')
        
        if plotname=="":
            plotname = "plot_network_"+str(dna["name"])+".png"
        
        if interactive:
            plt.show()
        else:    
            plt.savefig(plotname, dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format="png",
            transparent=False, bbox_inches='tight', pad_inches=0.1,
            frameon=None)
            plt.close()
            
            if add2mov:
                self.plotnames=self.plotnames+" "+plotname

    # =======================================================
    def plots2gif(self):
        os.system("convert -delay 50 "+self.plotnames+" "+"plot_network_evolve.gif")
    # =======================================================
    # =======================================================
    # Misc Functions
    # =======================================================
    # =======================================================
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    # =======================================================
    def constant(self,x,fac=1.0):
        return x*fac
    # =======================================================
    def prediction_error(self,dna,X,Y):
        Yhat = self.forward_propagation(dna,X)
        error=(Y - Yhat)
        cost_error = np.mean(np.abs(error)**2) 
        
        #------------------------------------
        nnodes=len(dna["node_id"])
        ncons=0
        nconws=0
        for n in range(len(dna["node_id"])):
            rnid = dna["node_id"][n]
            ncons+=len(dna["c"+str(rnid)][0])
            nconws+=np.sum(np.asarray(dna["c"+str(rnid)][1])**2)
        #------------------------------------
        cost_nodes=0.5*nnodes*self.costfac[0]
        cost_connections=0.5*ncons*self.costfac[1]
        cost_weights =0.5*nconws/(ncons+1e-20)*self.costfac[2]
        #------------------------------------
        
        cost=cost_error + cost_nodes+ cost_connections +cost_weights
        #------------------------------------
        dna["fitness"]=[cost]
        return cost
    # =======================================================
    def random_weight(self):
        return -self.wmax+(2*self.wmax)*np.random.rand()
    # =======================================================
    def random_weight_2(self,wamp):
        return 0.5*wamp+(wamp)*np.random.rand()
    # =======================================================
    def specie_ctr(self):
        self.speciectr+=1
        return self.speciectr
    # =======================================================
    def test_branin(self,a):
        x,y=a[:,0],a[:,1]
        z=np.zeros((a.shape[0],1))
        a,b,c=1,5.1/(4*np.pi**2),5/np.pi
        r,s,t=6,10,1/(8*np.pi)
        z[:,0]=a*(y-b*x**2+c*x-r)**2+s*(1-t)*np.cos(x)+s
        return z 
