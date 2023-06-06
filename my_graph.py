from copy import deepcopy
import sys
import numpy as np
import torch

def is_conjunction(p):
    p = p.split("#")[0][1:-1]
    ps = p.split(",")
    return len(ps)<2 or ps[0]==ps[1]

class Graph:

    num_feats = -1
    zeroFeats = None
    num_edges = 0
    num_edges_threshold = 0
    threshold = -1
    featIdx = 0

    def __init__(self, gpath, from_raw=True, args=None):
        self.args = args
        self.device = 'cuda:'+str(args.device) if (args and args.use_cuda>0) else 'cpu'
        if from_raw:
            self.buildGraphFromFile(gpath)
        # else:
        #     self.loadFromFile(gpath)

    def buildGraphFromFile(self,gpath):
        print ("gpath: ", gpath)
        if self.args and self.args.threshold:
            Graph.threshold = self.args.threshold
        f = open(gpath)
        Graph.featIdx = self.args.featIdx
        self.pred2idx = {}
        self.idxpair2score = {}
        self.allPreds = []
        first = True
        #read the lines and form the graph
        lIdx = 0
        isConj = False

        for line in f:
            line = line.replace("` ","").rstrip()
            lIdx += 1
            if first:
                self.name = line
                self.rawtype = line.split(",")[0].split(" ")[1]
                self.types = self.rawtype.split("#")
                if len(self.types)<2:
                    self.types = line.split(" ")[0].split("#")
                    self.rawtype = '#'.join(self.types)

                if len(self.types) == 2:
                    if self.types[0]==self.types[1]:
                        self.types[0] += "_1"
                        self.types[1] += "_2"

                first = False

            elif line == "":
                continue

            elif line.startswith("predicate:"):
                #time to read a predicate
                pred = line[11:]

                if (self.args.CCG and is_conjunction(pred)):
                    isConj = True
                    continue
                else:
                    isConj = False

                #The node
                if pred not in self.pred2idx:
                    self.pred2idx[pred] = len(self.pred2idx)
                    self.allPreds.append(pred)
                feat_idx = -1

            else:
                if (self.args.CCG and isConj):
                    # print "isConj"
                    continue
                if "num neighbors" in line:
                    continue
                #This means we have #cos sim, etc
                if line.endswith("sims") or line.endswith("sim"):
                    order = 0
                    feat_idx += 1
                    sim_name = line.lower()
                    # print ("line was: ", line)
                    #cos: 0, lin's prob: 1, etc

                else:
                    #Now, we've got sth interesting!
                    if feat_idx != Graph.featIdx:
                        continue
                    try:
                        ss = line.split(" ")
                        nPred = ss[0]
                        if self.args.CCG and Graph.is_conjunction(nPred):
                            continue
                        sim = ss[1]
                    except:
                        continue

                    order += 1

                    if self.args.maxRank and order>self.args.maxRank:
                        continue

                    if nPred not in self.pred2idx:
                        self.pred2idx[nPred] = len(self.pred2idx)
                        self.allPreds.append(nPred)
                    self.idxpair2score[(self.pred2idx[pred], self.pred2idx[nPred])] = float(sim)

        f.close()
        if first:
            line = 'types: '+gpath.split('/')[-1][:-8]+', num preds: 0'
            self.name = line
            self.rawtype = line.split(",")[0].split(" ")[1]
            self.types = self.rawtype.split("#")
            if len(self.types)<2:
                self.types = line.split(" ")[0].split("#")

            if len(self.types) == 2:
                if self.types[0]==self.types[1]:
                    self.types[0] += "_1"
                    self.types[1] += "_2"

    def loadFromFile(self, gpath):
        checkpoint = torch.load(gpath)
        # self.w_sparse = checkpoint['w_sparse']
        self.pred2idx = checkpoint['pred2idx']
        self.allPreds = checkpoint['allPreds']
        indices = checkpoint['w_sparse_indices']
        values = checkpoint['w_sparse_values']
        size = checkpoint['w_sparse_size']
        self.name, self.rawtype, self.types = checkpoint['names']
        return indices, values, size

    def dumpToFile(self, gpath, indices, values, size):
        # d = {'w_sparse_indices':self.w_sparse.indices(), 
        #     'w_sparse_values':self.w_sparse.values(),
        #     'w_sparse_size':self.w_sparse.size(),
        d = {'w_sparse_indices':indices, 
            'w_sparse_values':values,
            'w_sparse_size':size,
            'pred2idx':self.pred2idx, 
            'allPreds':self.allPreds,
            'names':(self.name, self.rawtype, self.types)}
        torch.save(d, gpath)

    def get_sparse_component(self):
        keys = self.idxpair2score.keys()
        indices = torch.LongTensor(list(zip(*keys)))
        values = torch.FloatTensor([self.idxpair2score[k] for k in keys])
        thesize = torch.Size([len(self.pred2idx), len(self.pred2idx)])
        if len(self.idxpair2score) == 0:
            indices = torch.empty([2,0]).long()
            values = torch.empty([0])
        return indices, values, thesize

    def composing(self, indices, values, thesize):
        self.w_sparse = torch.sparse_coo_tensor(indices, values, thesize, device=self.device).coalesce()
        try:
            del self.idxpair2score
        except:
            pass

    def set_Ws(self):
        indices, values, thesize = self.get_sparse_component()
        self.composing(indices, values, thesize)

    def writeGraphToFile(self,gpath):
        with open(gpath, 'w') as op:
            N = len(self.pred2idx)
            op.write(self.rawtype + "  type propagation num preds: " + str(N)+"\n")

            for pred in self.allPreds:
                scores = []

                thisMat = self.w_sparse
                predIdx = self.pred2idx[pred]
                thisRow = thisMat[predIdx].coalesce()
                neighs = thisRow.indices()[0]
                neighVals = thisRow.values()

                op.write("predicate: " + pred + "\n")
                op.write("max num neighbors: " + str(len(neighVals)) + "\n")
                op.write("\n")

                # print neighVals
                for i,neigh in enumerate(neighs):
                    pred2 = self.allPreds[neigh]
                    w = float(neighVals[i])
                    if Graph.threshold and w < Graph.threshold:
                        continue
                    scores.append((pred2,w))

                scores = sorted(scores,key = lambda x: x[1],reverse=True)

                op.write("global sims\n")
                s = ""
                for pred2,w in scores:
                    s += pred2 + " " + str(w)+"\n"
                op.write(s+"\n")

class dummyGraph:
    # only w_sparse_reserved
    def __init__(self, w_sparse):
        self.w_sparse = deepcopy(w_sparse)