import numpy as np
import json
import copy
import matplotlib.pyplot as plt

from activations import ACTIVATIONS

INNOVATIONSCTR, INNOVATIONS, GENOMCTR = 0, {}, 0
NODES = {}

### Genom class for neat ###
class Genom:

    ### Constructor ###
    def __init__(self, nids, nids_input, nids_output, structure, maxtimelevel=2,
                 _id=None, generation=0, parents=[None,None], crossover=True, iter_survived=0):

        ### Structural parameters ###
        self.nids = nids
        self.structure = structure
        self.maxtimelevel = maxtimelevel

        assert len(self.structure) == len(self.nids), "Length of structure must match length of Nids!"
        ### Previous nodestate ###
        self.last_states = []

        ### For prediction ###
        self.nids_input  = nids_input
        self.nids_output = nids_output
        self.nids_input_index  = [self.nids.index(nid) for nid in nids_input]
        self.nids_output_index = [self.nids.index(nid) for nid in nids_output]

        ### For evolution ###
        self._id = _id
        self.generation = generation
        self.parents = parents
        self.crossover = crossover
        self.iter_survived = iter_survived

    ### to Json ###
    def json(self):
        ### Note that the innvation manager is explicitly not added here! ###
        return {"_id": self._id, "nids": self.nids, "structure": self.structure, "maxtimelevel":self.maxtimelevel,
                "nids_input": self.nids_input, "nids_output": self.nids_output, "generation": self.generation,
                "parents": self.parents, "crossover": self.crossover, "iter_survived": self.iter_survived,
                }

    ### print detailed structure ###
    def show(self):
        return "<{}>".format(json.dumps(self.json(), indent=4, sort_keys=True))

    ### When printing out ###
    def __repr__(self):
        return "<Genom ID {}, parents {} ,gen. {}, survived {}>".format(self._id, self.parents, self.generation, self.iter_survived)

    ###
    @property
    def innovationNumbers(self):
        return sorted([ino for nid in self.nids for ino in self.structure[nid]["connections"]["innovations"]])

    @property
    def numberOfConnections(self):
        return sum([len(self.structure[nid]["connections"]["snids"]) for nid in self.nids])

    @property
    def sumOfWeightsAndBiases(self):
        return sum([sum([abs(w) for w in self.structure[nid]["connections"]["weights"]]) + abs(self.structure[nid]["bias"]) for nid in self.nids])

    ### ======================================
    # Innovation manager stuff
    ### ======================================
    ### Add innovation ###
    @staticmethod
    def _addInnovation(n1, n2, level):
        global INNOVATIONSCTR
        ### Name of the invention ###
        name = "{}-{}-{}".format(n1, n2, level)

        if name in list(INNOVATIONS.keys()):
            return INNOVATIONS[name]["id"]
        else:
            INNOVATIONSCTR += 1
            INNOVATIONS[name] = {"id":INNOVATIONSCTR, "feats": [n1, n2, level]}
            print("New innovationID {} ({}) added!".format(INNOVATIONSCTR,name))
            return INNOVATIONSCTR

    ### Return innovation features ###
    @staticmethod
    def _getFeatures(ino):
        return [vals for key, vals in INNOVATIONS.items() if vals["id"] == ino][0]["feats"]

    ### ======================================
    # Travesere Back ###
    ### ======================================
    def traverse_postorder(self, outputnode, alreadyrun=[]):
        nodes_postorder = []

        def recurse(node):
            if node in alreadyrun:
                return
            for snid in self.structure[node]["connections"]["snids"]:
                recurse(snid)

            nodes_postorder.append(node)
            alreadyrun.append(node)

        recurse(outputnode)
        return nodes_postorder

    ### ======================================
    # Feedforward run
    ### ======================================
    ### Make a forward prediction ###
    def run(self, X):

        ### Check input ###
        assert X.shape[1] == len(self.nids_input_index), "Input nodes must match dimesion of X!"

        ### Initialize node states ###
        node_states = np.zeros((X.shape[0], len(self.nids), self.maxtimelevel+1))

        ### Initialiaze previous node state for recurrent connections ###
        if len(self.last_states)>0:
            for t, last_state in enumerate(self.last_states):
                node_states[:,:,t+1] = last_state

        ### Assign values to input nodes ###
        node_states[:,self.nids_input_index, 0] = X

        ### Forward propagation ###
        for nid_index in range(len(self.nids_input), len(self.nids)):
            nid = self.nids[nid_index]
            snids  = self.structure[nid]["connections"]["snids"]
            weights  = self.structure[nid]["connections"]["weights"]
            active = self.structure[nid]["connections"]["active"]
            level = self.structure[nid]["connections"]["level"] # The time level of the connection

            bias = self.structure[nid]["bias"]
            fct = ACTIVATIONS[self.structure[nid]["activation"]]

            assert all(snid in self.nids for snid in snids), "Snid not in nid: {}, {}, {}".format(self.nids, self.structure, self.parents)
            snids_index = [self.nids.index(snid) for snid in snids]

            ### Calculate node state value ###
            if len(snids) > 0:
                #assert nid_index>max([snid for l, snid in zip(level, snids_index) if l == 0], default=0), "Network is not feedforward! nids: {}, structure:{}, parents {}".format(self.nids, self.structure, self.parents)
                node_states[:,nid_index, 0] += fct(np.sum(np.asarray(weights)* np.asarray(active)* node_states[:, snids_index, level], axis=1) + bias)

            ### The node seems not to have any input ###
            else:
                continue

        #print(node_states[:,:,0])
        ### Store node state (for recurrent connections) ###
        self.last_states.insert(0, node_states[:,:,0].copy())
        self.last_states = self.last_states[:self.maxtimelevel]

        ### Return output nodes values ###
        return node_states[:,self.nids_output_index, 0].reshape(X.shape[0],len(self.nids_output_index))

        ### Check input ###
        assert X.shape[1] == len(self.nids_input_index), "Input nodes must match dimesion of X!"

        ### Initialize node states ###
        node_states = np.zeros((X.shape[0], len(self.structure), self.maxtimelevel+1))
        nids = sorted(list(self.structure.keys()))

        ### Initialiaze previous node state for recurrent connections ###
        if len(self.last_states)>0:
            for t, last_state in enumerate(self.last_states):
                node_states[:,:,t+1] = last_state

        ### Assign values to input nodes ###
        node_states[:,[nids.index(nid) for nid in self.nids_input], 0] = X

        ### Go through graph ###
        alreadyrun = self.nids_input.copy()
        for nid_output in self.nids_output:

            for nid in self.traverse_postorder(nid_output, alreadyrun):

                snid_index = [nids.index(snid) for snid in self.structure[nid]["connections"]["snids"]]
                weights  = self.structure[nid]["connections"]["weights"]
                active = self.structure[nid]["connections"]["active"]
                level = self.structure[nid]["connections"]["level"] # The time level of the connection

                bias = self.structure[nid]["bias"]
                fct = ACTIVATIONS[self.structure[nid]["activation"]]

                ### Calculate node state value ###
                if len(snid_index) > 0:
                    #assert nid_index>max([snid for l, snid in zip(level, snids_index) if l == 0], default=0), "Network is not feedforward! nids: {}, structure:{}, parents {}".format(self.nids, self.structure, self.parents)
                    node_states[:,nids.index(nid), 0] += fct(np.sum(np.asarray(weights)* np.asarray(active)* node_states[:, snids_index, level], axis=1) + bias)

                ### The node seems not to have any input ###
                else:
                    continue

        #print(node_states[:,:,0])
        ### Store node state (for recurrent connections) ###
        self.last_states.insert(0, node_states[:,:,0].copy())
        self.last_states = self.last_states[:self.maxtimelevel]

        ### Return output nodes values ###
        return node_states[:,self.nids_output_index, 0].reshape(X.shape[0],len(self.nids_output_index))

    ### ======================================
    # Feedforward run
    ### ======================================
    ### Make a forward prediction ###
    def runOLD(self, X):

        ### Check input ###
        assert X.shape[1] == len(self.nids_input_index), "Input nodes must match dimesion of X!"

        ### Initialize node states ###
        node_states = np.zeros((X.shape[0], len(self.nids), self.maxtimelevel+1))

        ### Initialiaze previous node state for recurrent connections ###
        if len(self.last_states)>0:
            for t, last_state in enumerate(self.last_states):
                node_states[:,:,t+1] = last_state

        ### Assign values to input nodes ###
        node_states[:,self.nids_input_index, 0] = X

        ### Forward propagation ###
        for nid_index in range(len(self.nids_input), len(self.nids)):
            nid = self.nids[nid_index]
            snids  = self.structure[nid]["connections"]["snids"]
            weights  = self.structure[nid]["connections"]["weights"]
            active = self.structure[nid]["connections"]["active"]
            level = self.structure[nid]["connections"]["level"] # The time level of the connection

            bias = self.structure[nid]["bias"]
            fct = ACTIVATIONS[self.structure[nid]["activation"]]

            assert all(snid in self.nids for snid in snids), "Snid not in nid: {}, {}, {}".format(self.nids, self.structure, self.parents)
            snids_index = [self.nids.index(snid) for snid in snids]

            ### Calculate node state value ###
            if len(snids) > 0:
                #assert nid_index>max([snid for l, snid in zip(level, snids_index) if l == 0], default=0), "Network is not feedforward! nids: {}, structure:{}, parents {}".format(self.nids, self.structure, self.parents)
                node_states[:,nid_index, 0] += fct(np.sum(np.asarray(weights)* np.asarray(active)* node_states[:, snids_index, level], axis=1) + bias)

            ### The node seems not to have any input ###
            else:
                continue

        #print(node_states[:,:,0])
        ### Store node state (for recurrent connections) ###
        self.last_states.insert(0, node_states[:,:,0].copy())
        self.last_states = self.last_states[:self.maxtimelevel]

        ### Return output nodes values ###
        return node_states[:,self.nids_output_index, 0].reshape(X.shape[0],len(self.nids_output_index))


    ### ======================================
    # Initialization
    ### ======================================
    ### Create random structure ###
    @classmethod
    def initializeRandomly(cls, ninputs, noutputs, maxtimelevel=2, paddcon=0.8, paddnode=0.00, paddbias=0.01, pmutact=0.0, nrerun=None, output_activation=None):
        global GENOMCTR
        nids_input, nids_output = [x for x in range(0,ninputs)], [x for x in range(ninputs ,ninputs+noutputs)]
        nids = nids_input+nids_output
        nrerun = max([ninputs, noutputs]) if nrerun is None else nrerun

        output_activation = noutputs*[0] if output_activation is None else output_activation
        input_activation = ninputs*[0]

        assert len(output_activation) == noutputs, "Output activation must match number of outputs"

        structure= {}
        for nid,act in zip(nids, input_activation+output_activation):
            structure[nid] = {'connections':{"snids":[], "weights":[], "level":[], "innovations": [], "active":[]},'activation': act, 'bias': 0.0}

        ### Create genom ###
        genom = cls(nids, nids_input, nids_output, structure, maxtimelevel=maxtimelevel,
                     _id=GENOMCTR, generation=0, parents=['init'])

        for _ in range(nrerun):
            if np.random.rand() < paddcon:
                genom = Genom.mutate_add_connection(genom)
            if np.random.rand() < paddbias:
                pass #genom = Genom.mutate_bias(genom)
            if np.random.rand() < paddnode:
                genom = Genom.mutate_add_node(genom)
            if np.random.rand() < pmutact:
                pass #genom = Genom.mutate_activation(genom)

        while genom.numberOfConnections == 0:
            genom = Genom.mutate_add_connection(genom)

        GENOMCTR +=1
        return genom

    ### Mutate bias ###
    @classmethod
    def mutate_activation(cls, genom1, generation=None):
        nids = genom1.nids.copy()
        structure = copy.deepcopy(genom1.structure)
        nid_mut = nids[np.random.randint(len(genom1.nids_input), len(nids))]
        structure[nid_mut]["activation"] = np.random.randint(0,len(ACTIVATIONS))

        print("changed activation!!!!")
        return cls(nids=nids, structure=structure, nids_output=genom1.nids_output, nids_input=genom1.nids_input,
                   maxtimelevel=genom1.maxtimelevel, generation=generation, parents=[genom1._id, 'mutate_activation'], _id=genom1._id,
                   iter_survived=genom1.iter_survived, crossover=genom1.crossover)


    ### Mutate bias ###
    @classmethod
    def mutate_bias(cls, genom1, valueabs=1.0, pbigChange=0.1, generation=None):
        nids = genom1.nids.copy()
        structure = copy.deepcopy(genom1.structure)
        nid_mut = nids[np.random.randint(len(genom1.nids_input), len(nids))]
        if np.random.rand() < pbigChange:
            structure[nid_mut]["bias"] = -valueabs+2*valueabs*np.random.rand()
        else:
            structure[nid_mut]["bias"] += valueabs*np.random.normal()

       # structure[nid_mut]["bias"] = np.clip(structure[nid_mut]["bias"],-2,2)

        return cls(nids=nids, structure=structure, nids_output=genom1.nids_output, nids_input=genom1.nids_input,
                   maxtimelevel=genom1.maxtimelevel, generation=generation, parents=[genom1._id, 'mutate_bias'], _id=genom1._id,
                   iter_survived=genom1.iter_survived, crossover=genom1.crossover)

    ### Mutate weight ###
    @classmethod
    def mutate_weight(cls, genom1, valueabs=1.0, pbigChange=0.1, generation=None):
        nids = genom1.nids.copy()
        structure = copy.deepcopy(genom1.structure)
        nid_mut = nids[np.random.randint(len(genom1.nids_input), len(nids))]

        if len(structure[nid_mut]["connections"]["snids"]) > 0:
            index = np.random.randint(0, len(structure[nid_mut]["connections"]["snids"]))

            if np.random.rand() < pbigChange:
                structure[nid_mut]["connections"]["weights"][index] = -valueabs+2*valueabs*np.random.rand()
            else:
                structure[nid_mut]["connections"]["weights"][index] += valueabs*np.random.normal()

            #structure[nid_mut]["connections"]["weights"][index] = np.clip(structure[nid_mut]["connections"]["weights"][index],-2,2)

            return cls(nids=nids, structure=structure, nids_output=genom1.nids_output, nids_input=genom1.nids_input,
                       maxtimelevel=genom1.maxtimelevel, generation=generation, parents=[genom1._id, 'mutate_weight'], _id=genom1._id,
                       iter_survived=genom1.iter_survived, crossover=genom1.crossover)
        else:
            return genom1

    ### Add connection ###
    @classmethod
    def mutate_add_connection(cls, genom1, valueabs=1, pbigChange=0.1, maxretries=60, generation=None, timelevel=None):
        nids = genom1.nids.copy()
        structure = copy.deepcopy(genom1.structure)
        timelevel = genom1.maxtimelevel if timelevel is None else timelevel
        level = np.random.randint(0, timelevel)
        weight = valueabs*np.random.normal() if np.random.rand() > pbigChange else -valueabs+2*np.random.rand()

        #nid_add = nids[np.random.randint(len(genom1.nids_input), len(nids))]
        nid_add = nids[np.random.randint(len(genom1.nids_input), len(nids))]

        ### Find a valid SNID ###
        for _ in range(maxretries):
            snid = nids[np.random.randint(0, nids.index(nid_add))] if level == 0 else nids[np.random.randint(0, len(nids))]
            if not snid in structure[nid_add]["connections"]["snids"] and not snid in genom1.nids_output:
                break
            elif snid == nid_add and level >0:
                break
            if _ == maxretries -1:
                return genom1

        ### Add everything ###
        structure[nid_add]["connections"]["snids"].append(snid)
        structure[nid_add]["connections"]["weights"].append(weight)
        structure[nid_add]["connections"]["level"].append(level)
        structure[nid_add]["connections"]["active"].append(1)
        structure[nid_add]["connections"]["innovations"].append(Genom._addInnovation(snid, nid_add, level))

        return cls(nids=nids, structure=structure, nids_output=genom1.nids_output, nids_input=genom1.nids_input,
                   maxtimelevel=genom1.maxtimelevel, generation=generation, parents=[genom1._id, 'mutate_add_connection'], _id=genom1._id,
                   iter_survived=genom1.iter_survived, crossover=genom1.crossover)

    ### Remove node ###
    @classmethod
    def mutate_remove_connection(cls, genom1, generation=None):
        nids = genom1.nids.copy()
        structure = copy.deepcopy(genom1.structure)

        if len(genom1.nids_input) + len(genom1.nids_output) == len(nids):
            nid_dis = nids[len(genom1.nids_input)]
        else:
            nid_dis = nids[np.random.randint(len(genom1.nids_input), len(nids)-len(genom1.nids_output))]

        if len(structure[nid_dis]["connections"]["snids"]) > 1:
            idx = np.random.randint(0, len(structure[nid_dis]["connections"]["snids"]))
            structure[nid_dis]["connections"]["active"][idx] = 0 if structure[nid_dis]["connections"]["active"][idx] == 1 else 1

            return cls(nids=nids, structure=structure, nids_output=genom1.nids_output, nids_input=genom1.nids_input,
                       maxtimelevel=genom1.maxtimelevel, generation=generation, parents=[genom1._id, 'mutate_remove_connection'], _id=genom1._id,
                       iter_survived=genom1.iter_survived, crossover=genom1.crossover)
        else:
            return genom1

    ### Add node ###
    @classmethod
    def mutate_add_node(cls, genom1, activation=1, generation=None):
        nids = genom1.nids.copy()
        structure = copy.deepcopy(genom1.structure)

        ### get rnid index ###
        nid_new = 0
        while nid_new in nids:
            idx = np.random.randint(len(genom1.nids_input),len(genom1.nids))
            nid0, nid1 = genom1.nids[idx-1], genom1.nids[idx]
            nid_name = "{}-{}".format(nid0,nid1)
            if nid_name in list(NODES.keys()):
                nid_new = NODES[nid_name]
            else:
                nid_new = max(genom1.nids)
                while nid_new in genom1.nids + [nid for key, nid in NODES.items()]:
                    nid_new +=1
                NODES[nid_name] = nid_new

        ### Inserting new nid ###
        nids.insert(min([idx,len(genom1.nids)-len(genom1.nids_output)]), nid_new)
        structure[nid_new] = {"connections": {"snids":[], "innovations":[], "level":[], "weights":[], "active":[]}, "bias":0.0, "activation": 1}

        ### Replace connection with node ###
        if len(genom1.structure[nid1]["connections"]["snids"])>0:
            snid_idx = np.random.randint(0, len(genom1.structure[nid1]["connections"]["snids"]))
            snid = genom1.structure[nid1]["connections"]["snids"][snid_idx]
            weight = genom1.structure[nid1]["connections"]["weights"][snid_idx]
            level = genom1.structure[nid1]["connections"]["level"][snid_idx]
            ino  = genom1.structure[nid1]["connections"]["innovations"][snid_idx]
            active = genom1.structure[nid1]["connections"]["active"][snid_idx]
            ### Deactivate original connection ###
            structure[nid1]["connections"]["active"][snid_idx] = 0

        ### Add new node ###
        else:
            weight = np.random.normal()
            level = np.random.randint(0, genom1.maxtimelevel)
            activation = 1
            snid = nids[:nids.index(nid_new)][0]
            
        ### Add link to new node ###
        structure[nid_new]["connections"]["snids"].append(snid)
        structure[nid_new]["connections"]["level"].append(level)
        structure[nid_new]["connections"]["weights"].append(1.0)
        structure[nid_new]["connections"]["innovations"].append(Genom._addInnovation(snid, nid_new, level))
        structure[nid_new]["connections"]["active"].append(1.0)

        ### Add link to rnid ###
        structure[nid1]["connections"]["snids"].append(nid_new)
        structure[nid1]["connections"]["level"].append(level)
        structure[nid1]["connections"]["weights"].append(weight)
        structure[nid1]["connections"]["active"].append(1.0)
        structure[nid1]["connections"]["innovations"].append(Genom._addInnovation(nid_new, nid1, level))

        return cls(nids=nids, structure=structure, nids_output=genom1.nids_output, nids_input=genom1.nids_input,
                   maxtimelevel=genom1.maxtimelevel, generation=generation, parents=[genom1._id, 'mutate_add_node'], _id=genom1._id,
                   iter_survived=genom1.iter_survived, crossover=genom1.crossover)


    ### Remove node ###
    @classmethod
    def mutate_remove_node(cls, genom1, generation=None):
        nids = genom1.nids.copy()
        structure = copy.deepcopy(genom1.structure)

        ### Check if additional nodes exists ###
        if len(nids) > len(genom1.nids_input) + len(genom1.nids_output) +1:
            nid_remove = nids[np.random.randint(len(genom1.nids_input), len(nids)-len(genom1.nids_output))]
            snids = structure[nid_remove]["connections"]["snids"]
            level = structure[nid_remove]["connections"]["level"]
            weights = structure[nid_remove]["connections"]["weights"]
            active = structure[nid_remove]["connections"]["active"]
            ### Remove node ###
            nids.remove(nid_remove)
            del structure[nid_remove]

            ### Remove connections ###
            for nid in nids:
                if nid_remove in structure[nid]["connections"]["snids"]:
                    ### Removing connection to deleted node ###
                    index = structure[nid]["connections"]["snids"].index(nid_remove)
                    structure[nid]["connections"]["innovations"].pop(index)
                    structure[nid]["connections"]["weights"].pop(index)
                    structure[nid]["connections"]["snids"].pop(index)
                    structure[nid]["connections"]["level"].pop(index)
                    structure[nid]["connections"]["active"].pop(index)

                    ### Adding connections ###
                    for a,l,snid,weight in zip(active, level,snids,weights):
                        if not snid in structure[nid]["connections"]["snids"]:
                            structure[nid]["connections"]["innovations"].append(Genom._addInnovation(snid, nid, l))
                            structure[nid]["connections"]["weights"].append(weight)
                            structure[nid]["connections"]["snids"].append(snid)
                            structure[nid]["connections"]["level"].append(l)
                            structure[nid]["connections"]["active"].append(a)

    
            return cls(nids=nids, structure=structure, nids_output=genom1.nids_output, nids_input=genom1.nids_input,
                       maxtimelevel=genom1.maxtimelevel, generation=generation, parents=[genom1._id, 'mutate_remove_node'], _id=genom1._id,
                       iter_survived=genom1.iter_survived, crossover=genom1.crossover)

        else:
            return genom1

    ### Crossover genom 1 is assumed to be the fitter one ###
    @classmethod
    def crossover(cls, genom1, genom2, generation=None):
        global GENOMCTR

        n1, n2 = genom1.innovationNumbers, genom2.innovationNumbers
        cn = list(set(n1) & set(n2))

        excess = [ino for ino in n1 if ino>max(cn,default=0)] 
        disjoint = [ino for ino in n1 if ino<max(cn,default=0) and not ino in cn]
        n3 = cn + excess + disjoint

        ### Get NIDS ###
        cnids = list(set(genom1.nids) & set(genom2.nids))

        allnids = []
        for ino in n3:
            snid, rnid, level = Genom._getFeatures(ino)
            allnids.extend([snid,rnid])
        allnids = list(set(allnids))

        cnids = [cnids[idx] for idx in sorted(range(len([genom1.nids.index(nid) for nid in cnids])), key=[genom1.nids.index(nid) for nid in cnids].__getitem__)] 

        nids = cnids.copy()
        for genom in [genom1, genom2]:
            for nid in genom.nids:
                if nid in cnids or nid not in allnids:
                    continue

                for cnid in cnids:
                    cidx = genom.nids.index(cnid)
                    idx = genom.nids.index(nid)
                    if cidx > idx:
                        idx_insert = nids.index(cnid)
                        nids.insert(idx_insert, nid)
                        break


        ### add nids from innovation number
        structure = {}
        for nid in nids:
            if nid in genom1.nids_output+genom1.nids_input or nid in genom1.nids:
                structure[nid] = {'connections':{"snids":[], "weights":[], "level":[], "innovations": [], "active":[]},'activation': genom1.structure[nid]['activation'], 'bias': 0.0} 
            elif nid in genom2.nids:
                structure[nid] = {'connections':{"snids":[], "weights":[], "level":[], "innovations": [], "active":[]},'activation': genom2.structure[nid]['activation'], 'bias': 0.0}
            
        ### Add connections ###
        for ino in n3:
            snid, rnid, level = Genom._getFeatures(ino)

            ### Get weight, bias and activation ###
            if ino in cn:
                idx1 = genom1.structure[rnid]['connections']['snids'].index(snid)
                idx2 = genom2.structure[rnid]['connections']['snids'].index(snid)
                weight = [genom1.structure[rnid]['connections']['weights'][idx1], genom2.structure[rnid]['connections']['weights'][idx2]][np.random.randint(0,2)]
                bias = [genom1.structure[rnid]['bias'], genom2.structure[rnid]['bias']][np.random.randint(0,2)]
                active = [genom1.structure[rnid]['connections']['active'][idx1], genom2.structure[rnid]['connections']['active'][idx2]][np.random.randint(0,2)]
                activation = [genom1.structure[rnid]['activation'], genom2.structure[rnid]['activation']][np.random.randint(0,2)]

            elif ino in n1:
                idx1 = genom1.structure[rnid]['connections']['snids'].index(snid)
                weight = genom1.structure[rnid]['connections']['weights'][idx1]
                bias = genom1.structure[rnid]['bias']
                active = genom1.structure[rnid]['connections']['active'][idx1]
                activation = genom1.structure[rnid]['activation']

            elif ino in n2:
                idx2 = genom2.structure[rnid]['connections']['snids'].index(snid)
                weight = genom2.structure[rnid]['connections']['weights'][idx2]
                bias = genom2.structure[rnid]['bias']
                active = genom2.structure[rnid]['connections']['active'][idx2]
                activation = genom2.structure[rnid]['activation']

            structure[rnid]['connections']['snids'].append(snid)
            structure[rnid]['connections']['weights'].append(weight)
            structure[rnid]['connections']['level'].append(level)
            structure[rnid]['connections']['innovations'].append(ino)
            structure[rnid]['connections']['active'].append(active)
            structure[rnid]['bias'] = bias
            structure[rnid]['activation'] = activation

        if not len(nids) == len(structure):
            print(list(structure.keys()))
            print(nids)


        ### NIDS ###
        GENOMCTR+=1
        return cls(nids=nids, structure=structure, nids_output=genom1.nids_output, nids_input=genom1.nids_input,
                   maxtimelevel=genom1.maxtimelevel, generation=generation, parents=[genom1._id, genom2._id], _id=GENOMCTR,
                   crossover=True)

    ### =====================================
    # Calculate compability
    ### =====================================
    @staticmethod
    def compabilityMeasure(genom1, genom2, c1=1.0, c2=1.0, c3=0.2):
        ncons1 = max([genom1.numberOfConnections,1])
        ncons2 = max([genom2.numberOfConnections,1])
        nsumw1 = genom1.sumOfWeightsAndBiases
        nsumw2 = genom2.sumOfWeightsAndBiases

        n1, n2 = genom1.innovationNumbers, genom2.innovationNumbers
        cn = list(set(n1) & set(n2))

        excess = [ino for ino in n1 if ino>max(cn,default=0)] + [ino for ino in n2 if ino>max(cn,default=0)]
        disjoint = [ino for ino in n1 if ino<max(cn,default=0) and not ino in cn] + [ino for ino in n2 if ino<max(cn,default=0) and not ino in cn]
        N =  max([len(n1),len(n2),1])

        return c1*len(excess)/N + c2*len(disjoint)/N + c3*abs(nsumw1/ncons1-nsumw2/ncons2)



    ### Use to create a graph ###
    def showGraph(self, showLabels=True, store=False, picname="network.png", colors=["k", "gray","blue","m"]):
        executionLevel = []
        for nid in self.nids:
            if nid in self.nids_input:
                executionLevel.append(0)
            else:
                snids = self.structure[nid]["connections"]["snids"]
                executionLevel.append(max([executionLevel[self.nids.index(snid)] for snid in snids], default=0)+1)
        for nid in self.nids_output:
            executionLevel[self.nids.index(nid)] = max([executionLevel[self.nids.index(nido)] for nido in self.nids_output], default=0)

        y = []
        for i, (l, nid) in enumerate(zip(executionLevel, self.nids)):
            y.append(-0.5*executionLevel.count(l) + executionLevel[i:].count(l)+0.1-0.2*np.random.rand())

        for nid_index, nid in enumerate(self.nids):
            for active, snid, weight, level in zip(self.structure[nid]["connections"]["active"], self.structure[nid]["connections"]["snids"],
                self.structure[nid]["connections"]["weights"], self.structure[nid]["connections"]["level"]):
                snid_index = self.nids.index(snid)
                if active == 1:
                    color = "r" if weight>0 else "b"
                else:
                    color = "gray"
                style = "-" if level == 0 else "--"
                lw = np.clip(weight, -3,3)
                plt.plot([executionLevel[snid_index], executionLevel[nid_index]], [y[snid_index], y[nid_index]], ls=style, color=color, lw=lw)
   
        for nid_index, nid in enumerate(self.nids):
            plt.plot(executionLevel[nid_index], y[nid_index], 'o', color=colors[self.structure[nid]["activation"]], markersize=12)
            if showLabels:
                plt.text(executionLevel[nid_index], y[nid_index], nid, ha='center', va='center', color="w")

        plt.title("{}".format(self))
        plt.axis('off')
        if store:
            plt.savefig(picname)
            plt.close()
        else:
            plt.show()


#### TEST #######
if __name__ == "__main__":

    npop = 2
    genoms = [Genom.initializeRandomly(ninputs=2, noutputs=2, maxtimelevel=1) for pop in range(npop)]

    g1, g2 = genoms[0], genoms[1]
    g3, g4 = g1,g2

    for _ in range(8):
        g3 = Genom.mutate_add_node(g3)

    for _ in range(4):
        g4 = Genom.mutate_add_node(g4)

    for _ in range(7):
        g3, g4 = Genom.mutate_add_connection(g3), Genom.mutate_add_connection(g4)
 
    for _ in range(1):
        g3, g4 = Genom.mutate_remove_node(g3), Genom.mutate_remove_node(g4)

    g5 = Genom.crossover(g3,g4) 
   # g4 = Genom.mutate_add_node(g2)

    alreadyrun = g5.nids_input.copy()
    for nid in g5.nids_output:
        print(g5.traverse_postorder(nid,alreadyrun))

    # g3.showGraph()
    # g4.showGraph()
    g5.showGraph()