import sys
import numpy as np
sys.path.append("..")
from graph import graph
from lemma_baseline import qa_utils, berant
import evaluation.util
from lemma_baseline import baseline
from copy import deepcopy
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import os
from sklearn import svm, neural_network
from sklearn.metrics import precision_recall_curve
from sklearn import metrics
from constants.flags import opts
from ppdb import predict
import nltk
from pattern.en import conjugate, lemma, lexeme, PRESENT, SG, PL, PAST, PROGRESSIVE
from itertools import chain

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=np.nan)

#this is the default postfix, but can be overwritten by --sim_suffix
f_post_fix = "_sim.txt"

#only used for binary graphs (not used by default)
grPostfix = "_binc_lm1_.001_reg_1.5_1_.2_.01_i20_HTLFRG_confAvg_nodisc.txt"

try:
    conjugate(verb='gg',tense=PRESENT,aspect=PROGRESSIVE)
except:
    pass

with open('../../entgraph_gen/levy_types.txt','r') as f_type:
    types = list(f_type.readlines())
types = [t.strip().split('#') for t in types]

type2template = {}
type2templatelen = {}
wordtags = nltk.ConditionalFreqDist((w.lower(), t) for w, t in chain(nltk.corpus.brown.tagged_words(tagset="universal"), nltk.corpus.treebank.tagged_words(tagset="universal"))) 
with open('../../entgraph_gen/type_map.txt', 'r') as f:
    for l in f.readlines():
        ll = l.strip().split('\t')
        type2template[ll[0]] = ll[1]
        type2templatelen[ll[0]] = len(ll[1].split())

def min_match_length(A, B):
    for i in range(min(len(A), len(B))):
        if A[i]!=B[i]:
            return i
    return min(len(A),len(B))

can_negate = ['be', 'can', 'do', 'will', 'must', 'have', 'may', 'need', 'dare', 'ought']
def pred2template(pred, typeList, negated=False):
    # pred: like (sing.2,sing.to.start.of.2)#music#event
    if pred[:5] == 'NEG__':
        return pred2template(pred[5:], typeList, negated=True)
    thepos = pred.find('__')
    if thepos >= 0:
        return pred2template(pred[thepos+2:], typeList)
    try:
        preds, t1, t2 = pred.split('#')
        actor_1 = type2template[t1[:-2] if t1[-2]=='_' else t1]
        actor_2 = type2template[t2[:-2] if t2[-2]=='_' else t2]
        preds = preds[1:-1].split(',')
        preds = [p.split('.') for p in preds]
        reversing = False
        if typeList[0] == typeList[1]:
            if t1[-1] == '2' and t2[-1] == '1':
                reversing = True
        else:
            if typeList[0] == t2 and typeList[1] == t1:
                reversing = True
        if reversing:
            actor_1 += ' B'
            actor_2 += ' A'
        else:
            actor_1 += ' A'
            actor_2 += ' B'
        verbFlag = ('VERB' in wordtags[preds[0][0]] or preds[0][0] == "'s")
        orig_preds = [preds[0][:-1], preds[1][:-1]]
        if orig_preds == [["'s"],["of"]]:
            prefix = "Something's " if not negated else "Not something's " 
            return prefix + actor_1 + " of " + actor_2
        if not verbFlag:
            preds[0] = ['is'] + preds[0]
            preds[1] = ['is'] + preds[1]
        # assert len(set([p[0] for p in preds])) == 1
        A_active = (preds[0][-1] == '1')
        B_active = (preds[1][-1] == '1')
        minLength = min(len(preds[0]), len(preds[1]))-1
        pathway = (preds[0][:minLength] == preds[1][:minLength])

        # the only plural case
        if A_active and B_active:
            act = False
            if preds[0] == preds[1]:
                act = preds[0][:-1]
            elif pathway and len(preds[0])-1 == minLength:
                act = preds[1][:-1][::-1]
            elif pathway and len(preds[1])-1 == minLength:
                act = preds[0][:-1][::-1]
            if act:
                if act[0] in can_negate or not negated:
                    return actor_1 + ' and ' + actor_2 + conjugate(verb=act[0],tense=PRESENT,number=PL,negated=negated) + ' ' + ' '.join(act[1:])
                return actor_1 + ' and ' + actor_2 + conjugate(verb='do',tense=PRESENT,number=PL,negated=negated) + ' ' + ' '.join(act)
            else:
                print('pathway:', pathway, preds[0], preds[1], minLength)
                raise ValueError
        if preds[0][0] not in can_negate and negated:
            preds[0] = ['do'] + preds[0]
        if preds[1][0] not in can_negate and negated:
            preds[1] = ['do'] + preds[1]
        try:
            preds[0][0] = conjugate(verb=preds[0][0],tense=PRESENT,number=SG,negated=negated)
        except Exception as e:
            print(e, 'retrying')
            preds[0][0] = conjugate(verb=preds[0][0],tense=PRESENT,number=SG,negated=negated)
        preds[1][0] = conjugate(verb=preds[1][0],tense=PRESENT,number=SG,negated=negated)
        if A_active and not B_active:
            if pathway:
                verb = ' '.join(preds[1][:-1])
                if len(preds[0]) > len(preds[1]):
                    verb = ' '.join(preds[0][:-1])
                return actor_1 + ' ' + verb + ' ' + actor_2
            else:
                mml = min_match_length(preds[0], preds[1])
                return actor_1 + ' ' + ' '.join(preds[0][:-1]) + ' Something ' + ' '.join(preds[1][mml:-1]) + actor_2
        elif B_active and not A_active:
            mml = min_match_length(orig_preds[0], orig_preds[1])
            # maybe need phrase pos-tagging?
            if 'VERB' in wordtags[orig_preds[0][0]]:
                act = orig_preds[1][::-1][:-mml] + ['to'] + orig_preds[0]
            else:
                act = orig_preds[1][::-1] + orig_preds[0][mml:]
            if 'VERB' in wordtags[act[0]]:
                if negated and act[0] not in can_negate:
                    act = ['do'] + act
                act[0] = conjugate(verb=act[0],tense=PRESENT,number=SG, negated=negated)
            else:
                if negated:
                    act = ['is', 'not'] + act
                else:
                    act = ['is'] + act
            act = [actor_1] + act + [actor_2]
            return ' '.join(act)
        elif not A_active and not B_active:
            mml = min_match_length(orig_preds[0], orig_preds[1])
            if pathway:
                if negated:
                    be = 'is not'
                else:
                    be = 'is'
                act = [actor_1, be, conjugate(verb=orig_preds[0][0],tense=PAST,aspect=PROGRESSIVE,number=SG)] + orig_preds[0][1:] + orig_preds[1][mml:] + [actor_2]
                return ' '.join(act)
            else:
                if negated:
                    if orig_preds[0] not in can_negate:
                        orig_preds[0] = ['do'] + orig_preds[0]
                    orig_preds[0][0] = conjugate(verb=orig_preds[0][0],tense=PRESENT,number=SG,negated=negated)
                act = ['Something'] + orig_preds[0] + [actor_1] + orig_preds[1][mml:] + [actor_2]
                return ' '.join(act)
    except Exception as e:
        # logging.exception(e)
        # return pred2template(pred, reversing=reversing)
        print('error',e,pred)
        return 'NULL'

    print('unprocessable pred:', pred)
    return 'NULL'

def get_features(self,p,q):
    sent_p = pred2template(p, self.types)
    sent_q = pred2template(q, self.types)
    if p not in self.pred2Node or q not in self.pred2Node:
        # return None
        if sent_p not in self.sent2node or sent_q not in self.sent2node:
            return None

    if p==q:
        return np.ones(shape=(graph.Graph.num_feats))
    elif sent_p==sent_q:
        return np.ones(shape=(graph.Graph.num_feats))
    else:
        ret = np.zeros(shape=(graph.Graph.num_feats))
        if p in self.pred2Node and q in self.pred2Node:
            node1 = self.pred2Node[p]
            node2 = self.pred2Node[q]

            if node2.idx in node1.idx2oedges:
                sims = node1.idx2oedges[node2.idx].sims
                orders = node1.idx2oedges[node2.idx].orders
                ret[:len(sims)] = deepcopy(sims)
                ret[len(sims):] = orders
                return ret
        
        sentnode1 = self.sent2node[sent_p]
        sentnode2 = self.sent2node[sent_q]
        ret_cnt = 0
        for snode1 in sentnode1:
            for snode2 in sentnode2:
                if snode2.idx in snode1.idx2oedges:
                    sims = snode1.idx2oedges[snode2.idx].sims
                    orders = snode1.idx2oedges[snode2.idx].orders
                    ret[:len(sims)] += sims
                    ret[len(sims):] += orders
                    ret_cnt += 1
        if ret_cnt > 0:
            return ret / ret_cnt
        
        return graph.Graph.zeroFeats

def get_sum_simlar_feats(gr,p1s,q1s):
    feats = np.zeros(graph.Graph.num_feats)
    num_found = 0
    for p1 in p1s:
        for q1 in q1s:
            this_feats = get_features(gr,p1,q1)
            if this_feats is not None:
                if debug:
                    print ("not None in get_sims: ", p1,q1, this_feats)
                feats += this_feats
                num_found += 1
    if not num_found:
        feats = None
    else:
        feats /= num_found
    return feats

#deprecated function
#coef1 is for the case without embeddings. coef2 is with embeddings
#Propagating to t1, t2 from other types
# def get_coefs(p1,q1,p,q,t1,t2,a,is_typed,args):

def equalType(p1,t1,t2):
    p1t = p1.replace("_1", "").replace("_2", "")
    p1ss = p1t.split("#")
    ret = t1 == p1ss[1] and t2 == p1ss[2]
    if ret==False:
        if debug:
            print ("equalType false: ", p1,t1,t2)
    return ret

#p and q are the original ones. p1 and q1 are typed predicates
#It can be used for predPairFeats or predPairFeatsTyped
def add_feats_for_predPair(gr,p,q,a,t1,t2,p1,q1, p1s, q1s, predPairCounts, predPairSumCoefs, predPairSimSumCoefs, predPairFeats,predPairTypedExactFound,is_typed):

    feats = get_features(gr,p1,q1)
    feats_sim = get_sum_simlar_feats(gr,p1s,q1s)

    #This is for numNodes: num p, num q, num nodes

    if feats is not None or feats_sim is not None:
        no_exact_feats = False
        if debug:
            print ("feats:", feats)
            print ("feats_sim:", feats_sim)
        if feats is None:
            no_exact_feats = True
            feats = np.zeros(graph.Graph.num_feats)

        if is_typed:
            predPair = p+"#"+q+"#"+str(a)+"#"+t1+"#"+t2
            if debug:
                print ("featsT: ", feats)
        else:
            predPair = p+"#"+q+"#"+str(a)
        if debug:
            print ("predPair:", predPair)
            print ("p1,q1: ", p1, q1)
        if np.count_nonzero(feats)!=0 and not is_typed:
            if debug:
                print ("adding predPair to counts")
            evaluation.util.addPred(predPair, predPairCounts)
        if predPair not in predPairFeats:
            predPairFeats[predPair] = (np.zeros(num_feats))
            #For similarity ones:
            if not is_typed:
                if debug:
                    print ("setting predpairsums to zero")
                predPairSumCoefs[predPair] = 0
                predPairSimSumCoefs[predPair] = 0

        coef1,coef2 = 1, 1

        if is_typed and equalType(p1,t1,t2) and not no_exact_feats:
            if debug:
                print ("adding to predPairTyped: ", predPair)
            predPairTypedExactFound.add(predPair)

        if debug:
            print ("coef: ", coef1, " ", coef2)
        if is_typed:
            predPairFeats[predPair][0:graph.Graph.num_feats] = feats
            predPairFeats[predPair][graph.Graph.num_feats:2*graph.Graph.num_feats] = feats_sim

        else:
            if debug:
                print ("setting predPairFeats")
            predPairFeats[predPair][0:graph.Graph.num_feats] += coef1 * feats
            predPairFeats[predPair][graph.Graph.num_feats:2*graph.Graph.num_feats] += coef2 * feats_sim
            if not no_exact_feats:
                if debug:
                    print ("setting sumCoefs:")
                predPairSumCoefs[predPair] += coef1
            else:
                if debug:
                    print ("adding coef only for sims: ", coef2)
            predPairSimSumCoefs[predPair] += coef2
    return feats is not None and feats[0]!=0 and not no_exact_feats


def add_connectivity_for_predPair(gr,p,q,p1,q1,a,t1,t2,gholders,is_typed,predPairConnectedList,predPairConnectedWeightList,predPairTypedExactFound,args):
    if is_typed:
        predPair = p+"#"+q+"#"+str(a)+"#"+t1+"#"+t2
    else:
        predPair = p+"#"+q+"#"+str(a)

    coef = 0

    if p1 in gholders[0].pred2Node and q1 in gholders[0].pred2Node:
        coef = 1

        if is_typed and equalType(p1, t1, t2):
            if debug:
                print ("adding to predPairTyped: ", predPair)
            predPairTypedExactFound.add(predPair)

    for (idx,gholder) in enumerate(gholders):

        if is_connected_in_GHolder(gr,p1,q1,gholder):

            if debug:
                print ("typed based predpair:", predPair)
                print ("p1, q1:", p1, q1)
                print ("gr based coef: ", coef)
            predPairConnectedList[idx][predPair] = 1


def add_feats_for_unaryPair(gr, u, v, u1,v1, t, unaryPairFeats, unaryPairSumCoefs):
    feats_unary, coef1 = gr.get_features_unary(u1,v1)

    if feats_unary is not None:

        unaryPair = u + "#" + v + "#" + t
        if unaryPair not in unaryPairFeats:
            unaryPairFeats[unaryPair] = (np.zeros(graph.Graph.num_feats/2))#we don't have ranks for unary for simplicity

            unaryPairSumCoefs[unaryPair] = 0


        unaryPairFeats[unaryPair][0:graph.Graph.num_feats/2] += coef1 * feats_unary
        unaryPairSumCoefs[unaryPair] += coef1

    return feats_unary is not None

def form_samples_gr(gr, data, data_unary, predCounts, predPairCounts, predPairSumCoefs,predPairSimSumCoefs,predPairTypedSumCoefs, predPairTypedSimSumCoefs, predPairFeats, predPairFeatsTyped, predPairConnectedList,predPairConnectedWeightList, predPairTypedConnectedList, predPairTypedConnectedWeightList, predPairTypedExactFound, unaryPairFeatsTyped, unaryPairTypedSumCoefs, rels2Sims, gholders, args):
    if gr:
        types = gr.types
    else:
        types = gholders[0].types

    for (p,q,t1,t2,a) in data:
        ps = rels2Sims[p] if p in rels2Sims else [p]#The most similar ones
        qs = rels2Sims[q] if q in rels2Sims else [q]#The most similar ones
        #If you wanna exactly match the types, you should only take one of these!

        p1 = p + "#" + types[0] + "#" + types[1]
        p2 = p + "#" + types[1] + "#" + types[0]

        p1s = [pp + "#" + types[0] + "#" + types[1] for pp in ps]
        p2s = [pp + "#" + types[1] + "#" + types[0] for pp in ps]

        if a:
            q1 = q + "#" + types[0] + "#" + types[1]
            q2 = q + "#" + types[1] + "#" + types[0]

            q1s = [qq + "#" + types[0] + "#" + types[1] for qq in qs]
            q2s = [qq + "#" + types[1] + "#" + types[0] for qq in qs]

        else:
            q1 = q + "#" + types[1] + "#" + types[0]
            q2 = q + "#" + types[0] + "#" + types[1]

            q1s = [qq + "#" + types[1] + "#" + types[0] for qq in qs]
            q2s = [qq + "#" + types[0] + "#" + types[1] for qq in qs]

        if gholders:
            add_connectivity_for_predPair(gr,p,q,p1,q1,a,None,None,gholders,False,predPairConnectedList,predPairConnectedWeightList,None,args)
            add_connectivity_for_predPair(gr,p,q,p2,q2,a,None,None,gholders,False,predPairConnectedList,predPairConnectedWeightList,None,args)

        else:
            if p1 in gr.pred2Node or p2 in gr.pred2Node:
                evaluation.util.addPred(p, predCounts)

            if q1 in gr.pred2Node or q2 in gr.pred2Node:
                evaluation.util.addPred(q, predCounts)
            #gr,p,q,a,t1,t2,p1,q1, p1s, q1s, predPairCounts, predPairSumCoefs, predPairSimSumCoefs, predPairFeats,is_typed
            added = add_feats_for_predPair(gr,p,q,a,None,None,p1,q1,p1s,q1s,predPairCounts, predPairSumCoefs,predPairSimSumCoefs, predPairFeats,None,False)
            if added:
                if debug:
                    print ("added: ", p1, " ", q1," ", a, " ", types)

            added2 = add_feats_for_predPair(gr,p,q,a,None,None,p2,q2,p2s,q2s,predPairCounts, predPairSumCoefs,predPairSimSumCoefs, predPairFeats,None,False)
            if added2:
                if debug:
                    print ("added2: ", p2, " ", q2," ", a, " ", types)

        #Now, the typed ones!
        type_pair = None #This is to find the order of types (if any) that matches the graph!

        if t1==t2:
            if types[0][:-2]==types[1][:-2] and types[0][:-2]==t1:
                type_pair = types[0] + "#" + types[1]

        else:
            type_pair1 = t1+"#"+t2
            type_pair2 = t2+"#"+t1
            if type_pair1 == types[0] + "#" + types[1]:
                type_pair = t1+"#"+t2#We just check to see if graph types matches this predicate types. But still, should stick to t1, t2 not t2, t1!
            elif type_pair2 == types[0] + "#" + types[1]:
                type_pair = t1+"#"+t2


        if type_pair is not None:
            if debug:
                print ("a type match: ",type_pair, (p,q,t1,t2,a))

            this_types = type_pair.split("#")
            p1 = p + "#" + this_types[0] + "#" + this_types[1]
            p1s = [pp + "#" + this_types[0] + "#" + this_types[1] for pp in ps]
            if a:
                q1 = q + "#" + this_types[0] + "#" + this_types[1]
                q1s = [qq + "#" + this_types[0] + "#" + this_types[1] for qq in qs]
            else:
                q1 = q + "#" + this_types[1] + "#" + this_types[0]
                q1s = [qq + "#" + this_types[1] + "#" + this_types[0] for qq in qs]

            if gholders:
                add_connectivity_for_predPair(gr,p,q,p1,q1,a,t1,t2,gholders,True,predPairTypedConnectedList,None,predPairTypedExactFound,args)
            else:
                added = add_feats_for_predPair(gr,p,q,a,t1,t2,p1,q1,p1s,q1s,predPairCounts,None, None, predPairFeatsTyped,predPairTypedExactFound,True)
                if added:
                    if debug:
                        print ("added typed: ", p1, " ", q1," ", a, " ", types)
    if not gholders:
        ts = types  # This is to find the order of types (if any) that matches the graph!
        ts[0] = ts[0].replace("_1","").replace("_2","")
        ts[1] = ts[1].replace("_1","").replace("_2","")

        for (u, v, t) in data_unary:
            u1 = u + "#" + t
            v1 = v + "#" + t

            if t==ts[0] or t==ts[1]:
                added = add_feats_for_unaryPair(gr, u, v, u1,v1, t, unaryPairFeatsTyped, unaryPairTypedSumCoefs)
                # if added:
                #     print "added unary typed: ", u, v, t

#is_connected based on idx
def is_idx_connected_in_TNF(idx1,idx2,tnf):
    c1 = tnf.node2sccIdx[idx1]
    c2 = tnf.node2sccIdx[idx2]
    return c1 == c2 or c2 in tnf.scc.dedges[c1]

def is_connected_in_GHolder(gr,p1,q1,gholder):
    if p1 in gholder.pred2Node and q1 in gholder.pred2Node:
        if gr:
            idx1 = gr.pred2Node[p1].idx
            idx2 = gr.pred2Node[q1].idx
        else:
            idx1 = gholder.pred2Node[p1]
            idx2 = gholder.pred2Node[q1]

        tnf1_idx = gholder.node2comp[idx1]
        tnf2_idx = gholder.node2comp[idx2]

        if tnf1_idx != tnf2_idx:#They're already in different components, so there's no way for them to be connected
            return False
        else:
            tnf = gholder.TNFs[tnf1_idx]
            idx1 = tnf.idx2ArrayIdx[idx1]#We must see the index of the node in the subgraph! (Needed when we do decomposition)
            idx2 = tnf.idx2ArrayIdx[idx2]
            return is_idx_connected_in_TNF(idx1,idx2,tnf)

    else:

        return False

def form_samples(fnames,fnames_unary,orig_fnames,engG_dir_addr,fname_feats=None, rels2Sims=None, args=None):
    num_of_all_pps = 0
    if args and args.featIdx is not None:
        graph.Graph.featIdx = args.featIdx

    global num_feats
    num_feats = -1

    data_list = [evaluation.util.read_data(fnames[i], orig_fnames[i], args.CCG, args.typed, args.LDA) for i in range(len(fnames))]
    data_list_unary = []
    if fnames_unary:
        data_list_unary = [evaluation.util.read_data_unary(fnames_unary[i], args.typed) for i in range(len(fnames_unary))]

    predc_ds = []
    predPairc_DS = []
    predPaircTyped_DS = []
    predPairc_Pos_DS = []
    lmbdas = None

    for data in data_list:
        (predCDS_, predPairsCDS_, predPairsTypedCDS_, predPairsC_Pos_DS_) = evaluation.util.getPredPairs(data)
        predc_ds.append(predCDS_)
        predPairc_DS.append(predPairsCDS_)
        predPaircTyped_DS.append(predPairsTypedCDS_)
        predPairc_Pos_DS.append(predPairsC_Pos_DS_)
        num_of_all_pps += len(predPairsCDS_)

    predPairTypedExactFound = set()

    if fname_feats is not None:
        predPairFeats, predPairFeatsTyped, predPairSumCoefs,predPairTypedExactFound = evaluation.util.read_predPairFeats(fname_feats, data_list)
        num_feats = evaluation.util.num_feats
        return data_list, predPairSumCoefs, predPairFeats,predPairFeatsTyped, None, None, None, None, predPairTypedExactFound

    data_agg = []
    for data in data_list:
        for (p,q,t1s,t2s,_,a,_) in data:
            for t_i in range(len(t1s)):
                t1 = t1s[t_i]
                t2 = t2s[t_i]
                data_agg.append((p,q,t1,t2,a))
    data_agg = set(data_agg)

    data_agg_unary = []
    for data_unary in data_list_unary:
        data_agg_unary.extend(data_unary)
    data_agg_unary = set(data_agg_unary)

    if debug:
        print ("num unaryPairs: ", len(data_agg_unary))

    #The following to be filled based on the graphs!
    predCounts = {}
    predPairCounts = {}
    predPairSumCoefs = {}
    predPairSimSumCoefs = {}
    predPairTypedSumCoefs = {}
    predPairTypedSimSumCoefs = {}
    predPairFeats = {}
    predPairFeatsTyped = {}
    unaryPairFeatsTyped = {}
    unaryPairTypedSumCoefs = {}



    predPairConnectedList = None
    predPairConnectedWeightList = None
    predPairTypedConnectedList = None
    predPairTypedConnectedWeightList = None

    files = os.listdir(engG_dir_addr)
    files = list(np.sort(files))
    num_f = 0
    # print('len(files)', len(files),'read_sims', read_sims)

    for f in files:
        # if num_f == 100:#Use this for debugging!
        #     break
        if f[-4:] != '.txt':
            continue
        gpath=engG_dir_addr+f

        # print(type(f), str(f), (f_post_fix not in str(f)), os.stat(gpath).st_size)
        # if f_post_fix not in str(f) or os.stat(gpath).st_size == 0:
        #     print('continued')
        #     continue
        if debug:
            print ("fname: ", f)
        num_f += 1

        if num_f % 50 == 0:
            print ("num processed files: ", num_f)

        graphFilePath = gpath[:-8]+grPostfix

        if read_sims:
            gr = graph.Graph(gpath=gpath, args = args)
            gr.set_Ws()
            gr.pred2sent = {p: pred2template(p, gr.types) for p in gr.pred2Node}
            gr.sent2node = {}
            for p, s in gr.pred2sent.items():
                if s == 'NULL':
                    continue
                if s not in gr.sent2node:
                    gr.sent2node[s] = []
                gr.sent2node[s].append(gr.pred2Node[p])
        else:
            gr = None
        
        # print('num_feats', num_feats)
        if num_feats==-1 and read_sims:
            num_feats = 2*gr.num_feats
        # print('num_feats', num_feats, 'read_sims', read_sims)

        if read_sims:
            if debug:
                print ("gr size: ", sys.getsizeof(gr), "num edge: ", gr.num_edges)

        if debug:
            print ("reading TNFs: ")
        lIdx = 0
        if args.tnf:
            try:
                gLines = open(graphFilePath).readlines()
            except:
                if debug:
                    print ("exception to open: ", graphFilePath)
                continue

            gLines = [l.rstrip() for l in gLines]

            types = f[:-8].split("#")
            if debug:
                print ("types: ", types)
            if types[0] == types[1]:
                types[0] += "_1"
                types[1] += "_2"

            gholders = []
            lmbdaIdx = 0
            while lIdx<len(gLines):
                if debug:
                    print ("lmbdaIdx: ", lmbdaIdx)
                gholder = graph_holder.GHolder(gr,-1,gLines,lIdx,types)
                lIdx = gholder.lIdx
                gholder.clean()#Removes the pgraphs of its TNFs
                gholders.append(gholder)

            if lmbdas is None:

                lmbdas = []
                for gholder in gholders:
                    lmbdas.append(gholder.TNFs[0].lmbda)
                predPairConnectedList = [{} for _ in lmbdas]  # Is predPair connected in the graph?
                predPairConnectedWeightList = [{} for _ in lmbdas]
                predPairTypedConnectedList = [{} for _ in lmbdas]  # Is predPair typed connected in the graph?


                predPairTypedConnectedWeightList = None

            if debug:
                print ("TNFs read")
        else:
            gholders = None

        form_samples_gr(gr, data_agg, data_agg_unary, predCounts, predPairCounts, predPairSumCoefs, predPairSimSumCoefs, predPairTypedSumCoefs, predPairTypedSimSumCoefs,
                        predPairFeats, predPairFeatsTyped, predPairConnectedList,predPairConnectedWeightList,predPairTypedConnectedList,predPairTypedConnectedWeightList, predPairTypedExactFound,
                        unaryPairFeatsTyped, unaryPairTypedSumCoefs, rels2Sims,
                        gholders,args)
        
        del gr
        del gholders

    for (idx,fname) in enumerate(fnames):

        if debug:
            print ("stats for: " + fname)

            print ("num all preds: ", len(predc_ds[idx]))
            print ("num all predPairs: ", len(predPairc_DS[idx]))
            print ("num all pos predPairs: ", len(predPairc_Pos_DS[idx]))


            print ("num preds covered: ", np.count_nonzero( [(pred in predCounts) for pred in predc_ds[idx]] ))
            print ("num all predPairs covered: ", np.count_nonzero( [(predpair in predPairCounts) for predpair in predPairc_DS[idx]] ))
            print ("num pos predPairs covered: ", np.count_nonzero( [(predpair in predPairCounts) for predpair in predPairc_Pos_DS[idx]] ))
            print ("predCounts: ")

        for p in predc_ds[idx]:
            if p in predCounts:
                if debug:
                    print (p + "\t" + str(predCounts[p]) + "\t" + str(predc_ds[idx][p]))
            else:
                if debug:
                    print (p + "\t" + "0"  + "\t" + str(predc_ds[idx][p]))

        if debug:
            print ("predPairCounts_Pos: ")

        for r in predPairc_Pos_DS[idx]:
            if r in predPairCounts:
                if debug:
                    print (r + "\t" + str(predPairCounts[r]) + "\t" + str(predPairc_DS[idx][r]))
            else:
                if debug:
                    print (r + "\t" + "0"+ "\t" + str(predPairc_DS[idx][r]))

    if read_sims:

        predpair_set = []
        for predPairc in predPairc_DS:
            for predpair in predPairc:
                predpair_set.append(predpair)

        predpair_set = set(predpair_set)

        predpairTyped_set = []
        for predPairc in predPaircTyped_DS:
            for predpair in predPairc:
                predpairTyped_set.append(predpair)

        predpairTyped_set = set(predpairTyped_set)

        #divide the features!
        for r in unaryPairFeatsTyped:

            if unaryPairTypedSumCoefs[r] != 0:#In the above case, it won't be zero
                unaryPairFeatsTyped[r] /= unaryPairTypedSumCoefs[r]

        # divide the unary features!
        for r in predpair_set:
            if r in predPairSumCoefs:

                if predPairSumCoefs[r] != 0:  # In the above case, it won't be zero
                    predPairFeats[r][0:graph.Graph.num_feats] /= predPairSumCoefs[r]
                if predPairSimSumCoefs[r] != 0:
                    predPairFeats[r][graph.Graph.num_feats:2 * graph.Graph.num_feats] /= predPairSimSumCoefs[r]

        if debug:
            print ("predPairFeats: ")

        if not os.path.isdir('feats'):
            os.mkdir('feats')

        f_feats = open('feats/feats_'+args.method+'.txt','w')

        for r in predpair_set:
            if r in predPairSumCoefs:
                line = r + "\t" + str(predPairSumCoefs[r]) + "\t" + str(predPairSimSumCoefs[r]) + "\t" + str(predPairFeats[r])
                if debug:
                    print (line)
                f_feats.write(line+'\n')
            else:
                # print('num_feats', num_feats)
                predPairFeats[r] = deepcopy(np.zeros(num_feats))
                line = r + "\t" + "0\t0"+ "\t" + str(np.zeros(num_feats))
                if debug:
                    print (line)
                f_feats.write(line + '\n')

        if debug:
            print ("predPairFeatsTyped: ")

        for r in predpairTyped_set:
            if r in predPairFeatsTyped:
                line = r + "\t" + str(predPairFeatsTyped[r])
                if debug:
                    print (line)
                f_feats.write(line + '\n')
            else:
                predPairFeatsTyped[r] = deepcopy(np.zeros(num_feats))
                line = r + "\t" + str(np.zeros(num_feats))
                if debug:
                    print (line)
                f_feats.write(line + '\n')

        f_feats.write("predPairTypedExactFound:\n")
        for x in predPairTypedExactFound:
            f_feats.write(x + '\n')

        if debug:
            print ("predPairFeatsTyped: ", predPairTypedExactFound)

        # Writing unary features

        f_feats_unary = open('feats/feats_' + args.method + '_unary.txt', 'w')

        for r in unaryPairFeatsTyped:
            line = r + "\t" + str(unaryPairTypedSumCoefs[r]) + "\t" + str(unaryPairFeatsTyped[r])
            if debug:
                print (line)
            f_feats_unary.write(line + '\n')

    if debug:
        print ("predPairConnectedList in fit predict:", predPairConnectedList)

    print('num_of_all_pps', num_of_all_pps)
    return data_list, predPairSumCoefs, predPairFeats,predPairFeatsTyped,predPairConnectedList,predPairConnectedWeightList,predPairTypedConnectedList, predPairTypedConnectedWeightList, predPairTypedExactFound



def get_typed_feats(data,predPairFeatsTyped):
    X_typed = []
    for (p,q,t1s,t2s,probs,a,l) in data:
        this_X = np.zeros(num_feats)
        for i in range(len(t1s)):
            this_X += predPairFeatsTyped[p+"#"+q+"#"+str(a)+"#"+t1s[i]+"#"+t2s[i]]*probs[i]
        X_typed.append(this_X)
    return X_typed

#works in both unsupervised and supervised settings to return the entailment scores
def fit_predict(data_list, predPairFeats,predPairFeatsTyped,predPairConnectedList,predPairTypedConnectedList,predPairTypedExactFound,args):

    [data_train,data_dev] = data_list

    if predPairConnectedList is None:
        if debug:
            print ("predPairConnectedList is None")
        Y_dev_TNF = None
        Y_dev_TNF_typed = None
    else:

        Y_dev_TNF = [[] for _ in range(len(predPairConnectedList))]
        Y_dev_TNF_typed = [[] for _ in range(len(predPairConnectedList))]

        for x in range(len(predPairConnectedList)):

            for (i,(p,q,t1s,t2s,probs,a,_)) in enumerate(data_dev):
                predPair = p+"#"+q+"#"+str(a)
                if predPair in predPairConnectedList[x]:
                    Y_dev_TNF[x].append(predPairConnectedList[x][predPair])

                else:
                    Y_dev_TNF[x].append(0)

                typed_l = 0
                for t_i in range(len(t1s)):
                    t1 = t1s[t_i]
                    t2 = t2s[t_i]
                    predPairTyped = p+"#"+q+"#"+str(a)+"#"+t1+"#"+t2
                    if predPairTyped in predPairTypedConnectedList[x]:
                        this_l = predPairTypedConnectedList[x][predPairTyped] * probs[t_i]
                        typed_l += this_l

                if debug:
                    print ("testing, ", predPairTyped)
                # Not sure if we should not in predPairTypedExFound as for the thresholding way, we finally didn't.
                if args.backupAvg and predPairTyped not in predPairTypedExactFound:
                    if debug:
                        print ("not in exact")
                    if predPair in predPairConnectedList[x]:
                        if debug:
                            print ("in connected, ", x)
                        typed_l = predPairConnectedList[x][predPair]

                        if debug:
                            print ("now typed_l", typed_l)

                Y_dev_TNF_typed[x].append(typed_l)


    if not read_sims:
        return None,Y_dev_TNF,Y_dev_TNF_typed, None, None


    X_train = [list(deepcopy(predPairFeats[p+"#"+q+"#"+str(a)])) for (p,q,_,_,_,a,l) in data_train]

    X_train_typed = get_typed_feats(data_train,predPairFeatsTyped)

    if not args.useSims:
        X_train = [x[0:len(X_train[0])//2] for x in X_train]#only the first half!
        X_train_typed = [x[0:len(X_train_typed[0])//2] for x in X_train_typed]#only the first half!

    if debug:
        print ("here shape: ", np.array(X_train).shape)

    if not args.calcSupScores:
        [X_train[i].extend(X_train_typed[i]) for i in range(len(X_train))]
    else:
        X_train = X_train_typed

    #X: avg, avg_rank, avg_emb, avg_rank_emb, avg_typed, avg_rank_typed, avg_emb_typed, avg_rank_emb_typed

    X_dev = [list(deepcopy(predPairFeats[p+"#"+q+"#"+str(a)])) for (p,q,_,_,_,a,l) in data_dev]

    X_dev_typed = get_typed_feats(data_dev,predPairFeatsTyped)

    if not args.useSims:
        X_dev = [x[0:len(X_dev[0])//2] for x in X_dev]#only the first half!
        X_dev_typed = [x[0:len(X_dev_typed[0])//2] for x in X_dev_typed]#only the first half!

    if not args.supervised:
        if args.oneFeat:

            if not args.saveMemory:
                if not args.useSims:
                    f_idx = graph.Graph.featIdx
                else:
                    f_idx = graph.Graph.num_feats + graph.Graph.featIdx
                if args.rankFeats:
                    f_idx += graph.Graph.num_feats//2
                    if debug:
                        print ('new feat idx for rank: ', f_idx)
            else:
                if args.useSims:
                    f_idx = 2
                else:
                    f_idx = 0

            if not args.exactType:
                if not args.rankDiscount:
                    Y_dev_pred = [x[f_idx] for x in X_dev]
                else:
                    Y_dev_pred = [x[f_idx]*x[f_idx+graph.Graph.num_feats//2] ** .5 for x in X_dev]
            else:
                if not args.rankDiscount:
                    Y_dev_pred = [x[f_idx] for x in X_dev_typed]
                else:
                    Y_dev_pred = [x[f_idx] * x[f_idx + graph.Graph.num_feats // 2] ** .5 for x in X_dev_typed]

                if args.backupAvg:
                    if not args.rankDiscount:
                        Y_dev_pred_backup = [x[f_idx] for x in X_dev]
                    else:
                        # this is first average, then multiplied, but should be the other way around
                        Y_dev_pred_backup = [x[f_idx] * x[f_idx + graph.Graph.num_feats / 2] ** .5 for x in X_dev]
                    Y_dev_pred2 = []
                    for i in range(len(Y_dev_pred)):
                        l = Y_dev_pred[i]

                        (p, q, t1s, t2s, probs, a, _) = data_dev[i]

                        if l==0:#In practice, if it's zero, it's zero for everything!!! I tested this!
                            l = Y_dev_pred_backup[i]

                        Y_dev_pred2.append(l)

                    Y_dev_pred = Y_dev_pred2

            if debug:
                print ("nnz Y_dev_pred: ", np.count_nonzero(Y_dev_pred))

        elif args.wAvgFeats:

            assert not args.typed#Because it's not implemented yet!

            ss = args.wAvgFeats.split()
            idxes = [np.int(x) for i,x in enumerate(ss) if i%2==0 ]
            weights = [np.float(x) for i, x in enumerate(ss) if i % 2 == 1]
            sum_weighs = sum(weights)

            def weighted_sum(x,idxes,weights):
                ret = 0
                for i,idx in enumerate(idxes):
                    ret += x[idx]*weights[i]
                return ret/sum_weighs

            Y_dev_pred = [weighted_sum(x,idxes,weights) for x in X_dev]

        elif args.gAvgFeats:

            assert not args.typed#Because it's not implemented yet!

            ss = args.wAvgFeats.split()
            idxes = [np.int(x) for i,x in enumerate(ss) if i%2==0 ]
            weights = [np.float(x) for i, x in enumerate(ss) if i % 2 == 1]
            sum_weighs = sum(weights)

            def weighted_sum(x,idxes,weights):
                ret = 0
                for i,idx in enumerate(idxes):
                    ret += x[idx]*weights[i]
                return ret/sum_weighs

            Y_dev_pred = [weighted_sum(x,idxes,weights) for x in X_dev]

        else:
            raise Exception("terrible exception")

        return Y_dev_pred,Y_dev_TNF,Y_dev_TNF_typed, None, None


    if not args.calcSupScores:
        [X_dev[i].extend(X_dev_typed[i]) for i in range(len(X_dev))]
    else:
        X_dev = X_dev_typed
    # X_dev = X_dev_typed

    Y_train = [l for (_,_,_,_,_,_,l) in data_train]
    Y_dev = [l for (_,_,_,_,_,_,l) in data_dev]

    if debug:
        print ("computing train pair recall: ")
    pair_recall = evaluation.util.compute_pair_recalls(X_train, Y_train)

    if debug:
        print ("pair recall: ", pair_recall)
        print ("computing dev pair recall: ")
    pair_recall = evaluation.util.compute_pair_recalls(X_dev, Y_dev)
    if debug:
        print ("pair recall: ", pair_recall)

    X_train = np.array(X_train)
    X_dev = np.array(X_dev)

    Y_train = np.array(Y_train)

    if debug:
        print ("final shape: ", X_train.shape, X_dev.shape)

    scaler = preprocessing.StandardScaler().fit(X_train)

    X_train = scaler.transform(X_train)
    X_dev = scaler.transform(X_dev)

    # changes made to build the supervised graph: let the code do what it was doing before, but just return a classifier
    # to build the classifier for the entailment graph building, make sure to use: sup 1 exactFeats and rankFeats...
    # cl = svm.SVC(probability=True)
    cl = neural_network.MLPClassifier(solver='adam', hidden_layer_sizes=(100), random_state=114514)

    # other options for the classifer below:
    # cl = svm.SVC(C=1.0,kernel='rbf',probability=True)
    # cl = LogisticRegression(penalty='l1')

    cl.fit(X_train,Y_train)

    if isinstance(cl,LogisticRegression) or isinstance(cl,neural_network.MLPClassifier):
        if debug:
            print ("lr coefs: ", cl.coef_)
        Y_dev_pred = cl.predict_proba(X_dev)
    else:
        Y_dev_pred = cl._predict_proba(X_dev)

    Y_dev_pred = [y for (_,y) in Y_dev_pred]
    if debug:
        print ("Y_dev_pred: ", Y_dev_pred)

    Y_train_pred = cl.predict(X_train)
    if debug:
        print ("train evaluation: ")
    eval(Y_train_pred,Y_train)

    if debug:
        print ("dev eval: ")
    eval([y>.5 for y in Y_dev_pred],Y_dev)

    return Y_dev_pred,Y_dev_TNF,Y_dev_TNF_typed, scaler, cl

def eval(Y_pred,Y, write = True):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(Y)):
        if Y_pred[i]==Y[i]:
            if Y_pred[i]==1:
                TP += 1
            else:
                TN += 1
        else:
            if Y_pred[i]==1:
                FP += 1
            else:
                FN += 1

    if write:
        print ("TP: ", TP)
        print ("FP: ", FP)
        print ("TN: ", TN)
        print ("FN: ", FN)

    if (TP+FP)==0 or TP==0:
        pr=0
        rec=0
        f1=0
    else:
        pr = np.float(TP)/(TP+FP)
        rec = np.float(TP)/(TP+FN)
        f1 = 2*pr*rec/(pr+rec)

    acc = np.float(TP+TN)/len(Y)

    if write:
        print ("prec: ", pr)
        print ("rec: ", rec)
        print ("f1: ", f1)
        print ("acc: ", acc)

        print (str(TP)+"\t"+str(FP)+"\t"+str(TN)+"\t"+str(FN)+"\t"+str(pr)+"\t"+str(rec)+"\t"+str(f1))

    return pr,rec, acc

#It will use Y_dev_base and Y_dev_pred0 to form Y_dev_pred which will be used to find precisions and recalls
#It will also change Y_dev_TNFs so it will be used later
def final_prediction(data_dev, data_dev_CCG, predPairFeats, predPairFeatsTyped, predPairSumCoefs, Y_dev_base, Y_dev_pred0,Y_dev_TNF0,Y_dev_TNF_typed0, lines_dev, root, args):

    Y_dev_pred = []
    Y_dev_base_const = []
    Y_dev = [l for (_,_,_,_,_,_,l) in data_dev]
    Y_dev_seen = []
    Y_dev_pred_seen = []

    if not args.no_lemma_baseline:
        print ("lemma baseilne eval:")
        eval(Y_dev_base,Y_dev)

    # f_out_predpair_seen = open('predpair_seen.txt', 'w')

    if orig_fnames[1]:

        if not args.berDS and not args.berDS_v2 and not args.berDS_v3:
            Y_dev_berant = berant.predict_Berant(root+"berant/",orig_fnames[1])
            Y_dev_berant2 = berant.predict_Berant(root+"berant/",orig_fnames[1],resource2=True)
        else:
            Y_dev_berant = berant.predict_Berant(root + "berant/", orig_fnames[1], context=True, berDS=True)
            Y_dev_berant2 = berant.predict_Berant(root + "berant/", orig_fnames[1], resource2=True,context=True, berDS=True)
        Y_dev_ppdb_xl = predict.predict(root+'ent/all_comb_ppdb_xl.txt',orig_fnames[1],args)
        Y_dev_ppdb_xxxl = predict.predict(root + 'ent/all_comb_ppdb_xxxl.txt',orig_fnames[1],args)

        print ("berant dev eval:")
        eval(Y_dev_berant,Y_dev)

        print ("berant dev2 eval:")
        eval(Y_dev_berant2,Y_dev)

        print ("ppdb_xl dev eval (valid only for levy or berant):")
        eval(Y_dev_ppdb_xl, Y_dev)

        print ("ppdb_xxxl dev eval (valid only for levy or berant):")
        eval(Y_dev_ppdb_xxxl, Y_dev)

        print ("analyze ppdb")
        for i, y in enumerate(Y_dev):
            if Y_dev_ppdb_xxxl[i] != Y_dev[i]:
                if debug:
                    print ("ppdb xxxl wrong: ", lines_dev[i])


    #Now do the final prediction!

    for (idx, _) in enumerate(data_dev):

        cl_used = False

        (p_ccg,q_ccg,_,_,_,a_ccg,_) = data_dev_CCG[idx]

        if not args.no_lemma_baseline and Y_dev_base[idx]:
            Y_dev_base_const.append(True)
            pred = True
            if debug:
                print ("lemma baseline: ", pred, lines_dev[idx])
        elif not args.no_constraints and a_ccg and qa_utils.constraint_y(p_ccg,q_ccg):
            pred = True
            if debug:
                print ("con_y: ", lines_dev[idx])
            Y_dev_base_const.append(True)
        elif not args.no_constraints and ((a_ccg and qa_utils.constraint_n(p_ccg,q_ccg)) or qa_utils.transitive_reverse(p_ccg,q_ccg,a_ccg)):
            if debug:
                print ("con_n: ", lines_dev[idx])
            Y_dev_base_const.append(False)
            pred = False
        else:
            Y_dev_base_const.append(False)
            if Y_dev_pred0:
                pred = Y_dev_pred0[idx]
            else:
                pred = False
            cl_used = True

        Y_dev_pred.append(pred)

        predPair = p_ccg+"#"+q_ccg+"#"+str(a_ccg)
        if predPairSumCoefs:
            predPairSeen = (predPair in predPairSumCoefs and predPairSumCoefs[predPair] > 0)
            # if predPairSeen:
            #     f_out_predpair_seen.write('1\n')
            # else:
            #     f_out_predpair_seen.write('0\n')
            if debug:
                print ("is seen: ", predPair, predPairSeen)
            if predPairSeen:
                Y_dev_pred_seen.append(pred)
                Y_dev_seen.append(Y_dev[idx])

        if not cl_used:
            Y_dev_berant[idx] = pred
            Y_dev_berant2[idx] = pred
            Y_dev_ppdb_xl[idx] = pred
            Y_dev_ppdb_xxxl[idx] = pred

            if Y_dev_TNF0 is not None:
                for i in range(len(Y_dev_TNF0)):
                    Y_dev_TNF0[i][idx] = pred
                    Y_dev_TNF_typed0[i][idx] = pred

    if not args.no_lemma_baseline:
        print ("eval baseline constraint: ")
        eval(Y_dev_base_const,Y_dev)

    print ("Berant final: ")
    eval(Y_dev_berant,Y_dev)

    print ("Berant2 final: ")
    eval(Y_dev_berant2,Y_dev)

    print ("ppdb_xl final: ")
    eval(Y_dev_ppdb_xl, Y_dev)

    print ("ppdb_xxxl final: ")
    eval(Y_dev_ppdb_xxxl, Y_dev)

    print ("analyze Berant")
    for i,y in enumerate(Y_dev):
        if Y_dev_berant2[i]!=Y_dev[i]:
            if debug:
                print ("Berant2's wrong: ", lines_dev[i])

    for (i,y) in enumerate(Y_dev_pred):
        if debug:
            print (lines_dev[i])
            print (y, " ", Y_dev[i])

    fpr, tpr, thresholds = metrics.roc_curve(Y_dev, Y_dev_pred)
    if args.write:
        s1 = root + out_dir
        #uncomment s2 and op_tp_fp if you want to have _roc.txt file
        # s2 = root + out_dir
        if not os.path.isdir(s1):
            os.mkdir(s1)
        # if not os.path.isdir(s2):
        #     os.mkdir(s2)
        op_pr_rec = open(s1 + args.method + ".txt",'w')
        op_Y_pred = open(s1 + args.method + "_Y.txt",'w')
        # op_tp_fp = open(s2 + args.method + "_roc.txt",'w')

    auc_fpr_tpr = metrics.auc(fpr, tpr)
    if debug:
        print ("auc_fpr_tpr: ", auc_fpr_tpr)

    auc = auc_fpr_tpr
    if args.write:
        op_pr_rec.write("auc fpr tpr: "+str(auc_fpr_tpr)+"\n")


    if debug:
        print ("tpr, fpr: ")
    for i in range(len(tpr)):
        try:
            if debug:
                print (tpr[i], " ", fpr[i], thresholds[i])
        except:
            pass

    if debug:
        print ("num seen: ", len(Y_dev_seen), " vs ", len(Y_dev))

    (precision, recall, thresholds) = precision_recall_curve(Y_dev, Y_dev_pred)
    # try:
    main_auc = evaluation.util.get_auc(precision[:-1], recall[:-1])
    # except:
    #     print('get auc failed!')
    #     main_auc = 0

    if args.write:
        op_pr_rec.write("auc: "+str(main_auc)+"\n")

    if debug:
        print ("main_auc:", main_auc)
    for i in range(len(Y_dev)):
        y_pred = Y_dev_pred[i]
        if y_pred==True:
            y_pred = 1
        elif y_pred==False:
            y_pred = 0
        op_Y_pred.write(str(Y_dev[i])+" "+str(y_pred)+"\n")
    # util.get_confidence_interval(Y_dev, Y_dev_pred)
    if debug:
        print ("avg pr score")
    a = metrics.average_precision_score(Y_dev,Y_dev_pred)
    if debug:
        print (a)
    b = metrics.average_precision_score(Y_dev, Y_dev_pred,average='micro')
    if debug:
        print (b)

    if debug:
        print ("auc: ", auc)

    prs_high = []
    recs_high = []

    threshold = .16 #For Happy classification :) #But it will be set to threshold for precision ~ .76
    threshold_set = False
    if debug:
        print ("pr_rec:")
    for i in range(len(precision)):

        if args.write:
            if i>0:
                op_pr_rec.write(str(precision[i])+ " "+ str(recall[i])+"\n")
                if precision[i] > .5 and precision[i]!=1:
                    prs_high.append(precision[i])
                    recs_high.append(recall[i])
        try:
            if not threshold_set and precision[i]>.748 and precision[i]<.765:
            # if not threshold_set and precision[i] > .85 and precision[i] < .86:
                threshold = thresholds[i]
                threshold_set = True
            if debug:
                print (precision[i], " ", recall[i], thresholds[i])
        except:
            if debug:
                print ("exception: ", precision[i],recall[i])
            pass


    if debug:
        print ("threshold set to: ", threshold)
    Y_dev_pred_binary = [y>threshold for y in Y_dev_pred]

    all_FPs = []
    all_FNs = []

    sample_f = open('samples.txt','w')

    if debug:
        if read_sims or args.instance_level:
            if predPairFeatsTyped:
                X_dev_typed = get_typed_feats(data_dev,predPairFeatsTyped)

            print ("results:")
            for (idx,(p,q,t1s,t2s,probs,a,l)) in enumerate(data_dev):
                line_info = lines_dev[idx]+"\t"+p+"#"+q+"#"+str(a)+"\t"+str(t1s)+"#"+str(t2s)+"\t"
                if predPairFeats:
                    line_info += str(predPairFeats[p + "#" + q + "#" + str(a)]) + "\t" + str(X_dev_typed[idx]) + "\t" +\
                                 str(Y_dev_pred[idx])
                if Y_dev_pred_binary[idx] and Y_dev[idx]:
                    conf_l = "TP"
                elif Y_dev_pred_binary[idx] and not Y_dev[idx]:
                    all_FPs.append(line_info)
                    conf_l = "FP"
                elif not Y_dev_pred_binary[idx] and Y_dev[idx]:
                    all_FNs.append(line_info)
                    conf_l = "FN"
                else:
                    conf_l = "TN"


                print (conf_l + " : " + lines_dev[idx])
                print ("pred: ", Y_dev_pred[idx])
                if predPairFeats:
                    predPair = p+"#"+q+"#"+str(a)
                    print (predPairFeats[predPair])
                    # print predPairFeatsTyped[p+"#"+q+"#"+str(a)+"#"+t1+"#"+t2]
                    print (X_dev_typed[idx])
                    if conf_l=="FN" and predPairSumCoefs and (predPair not in predPairSumCoefs or predPairSumCoefs[predPair]==0):
                        print ("unseen and FN")
                print (p+"#"+q+"#"+str(a))
                print (str(t1s)+"#"+str(t2s)+"\n")
                if args.LDA:
                    print (probs)

            print ("ours vs Berant's")
            for (idx,_) in enumerate(data_dev):
                if Y_dev_pred_binary[idx]!=Y_dev_berant[idx]:
                    if Y_dev_pred_binary[idx] == Y_dev[idx]:
                        print ("ours is correct: ", lines_dev[idx])
                    else:
                        print ("Berant's is correct: ", lines_dev[idx])


    print ("ours final: ")
    eval(Y_dev_pred_binary,Y_dev)

    # print "samples FPs:"
    # FPs = np.random.choice(all_FPs, 100, replace=False)
    # for l in FPs:
    #     print l
    #     sample_f.write(l+"\n")

    # print "samples FNs:"
    # FNs = np.random.choice(all_FNs, 100, replace=False)
    # for l in FNs:
    #     print l
    #     sample_f.write(l + "\n")

    return Y_dev_pred, Y_dev_berant

#Y_dev should be score
def write_detailed_res(data,lines,Y,Y_pred, Y_berant, base_threshold = .75):

    (precision, recall, thresholds) = precision_recall_curve(Y, Y_pred)

    threshold = .15
    for j in range(len(precision)):
        if precision[j] > base_threshold and precision[j] < base_threshold+.01:
            try:
                threshold = thresholds[j]
            except:
                pass
        print (str(precision[j]) + " " + str(recall[j]))
    print ("threshold is: ", str(threshold))

    Y_pred_binary = [y > threshold for y in Y_pred]

    write_detailed_res_binary(data,lines,Y,Y_pred,Y_pred_binary, Y_berant)


def write_detailed_res_binary(data,lines,Y,Y_pred,Y_pred_binary, Y_berant):
    for (idx, (p, q, t1s, t2s, probs, a, l)) in enumerate(data):
        if Y_pred_binary[idx] and Y[idx]:
            conf_l = "TP"
        elif Y_pred_binary[idx] and not Y[idx]:
            conf_l = "FP"
        elif not Y_pred_binary[idx] and Y[idx]:
            conf_l = "FN"
        else:
            conf_l = "TN"

        print (conf_l, ":", lines[idx])
        print ("pred: ", Y_pred[idx])
        print (p + "#" + q + "#" + str(a))
        print (str(t1s) + "#" + str(t2s) + "\n")

    print ("ours vs Berant's")
    for (idx, _) in enumerate(data):
        if Y_pred_binary[idx] != Y_berant[idx]:
            if Y_pred_binary[idx] == Y[idx]:
                print ("ours is correct: ", lines[idx])
            else:
                print ("Berant's is correct: ", lines[idx])

def calcGraphScores(gr, cl, scaler):

    fout = open(gpath[:-len(f_post_fix)] + "_sup.txt", 'w')

    N = len(gr.nodes)
    fout.write("types: " + gr.types + ", num preds: " + N + "\n")

    for i in range(N):
        node1 = gr.nodes[i]
        p = node1.id
        fout.write("predicate: " + p + "\n")

        fout.write("num neighbors: " + str(len(node1.oedges)) + "\n\n")
        fout.write("sup sim")

        for oedge in node1.oedges:
            idx2 = oedge.idx
            node2 = gr.nodes[idx2]
            q = node2.id

            feats = get_features(gr, p, q)
            feats = np.array(feats)
            feats = scaler.transform(feats)

            s = cl.predict_proba(feats)

            fout.write(q + " " + str(s) + "\n")
        fout.write("\n")

#These (until parameters) are fixed and won't change (too much)!

root = "../../gfiles/"
sysargs = sys.argv[1:]
args = opts(sysargs)

debug = graph.debug = qa_utils.debug = baseline.debug = evaluation.util.debug = predict.debug = berant.debug = args.debug
if args.tnf:
    from graph import graph_holder, gt_Graph, sccGraph
    graph_holder.debug = sccGraph.debug = gt_Graph.debug = debug

if args.outDir:
    out_dir = args.outDir+"/"
else:
    out_dir = 'results/pr_rec/'

if args.sim_suffix:
    f_post_fix = args.sim_suffix

assert (not args.dev or not args.test)
assert not args.snli or args.CCG
assert not (args.rankDiscount and args.rankFeats)

if not args.snli:
    #we only use unary for the snli experiments
    fnames_unary = None

if args.tnf:
    f_post_fix = "_sim.txt"#Won't use the file, just used to build the other address name!

if args.dev:
    if args.LDA:
        fnames_CCG = [root+"ent/train_new_LDA15rels.txt",root+"ent/dev_new_LDA15rels.txt"]
    else:
        if args.origLevy:
            fnames_CCG = [root + "ent/all_comb_rels.txt", root + "ent/dev1_rels_l8.txt"]
            fnames_oie = [root + "ent/all_rels_oie.txt", root + "ent/dev1_rels_oie.txt"]
            orig_fnames = [root + "ent/all_comb.txt", root + "ent/dev1.txt"]
        else:
            fnames_CCG = [root + "ent/all_comb_rels.txt", root + "ent/dev_rels.txt"]
            fnames_oie = [root + "ent/all_rels_oie.txt", root + "ent/dev_rels_oie.txt"]
            orig_fnames = [root + "ent/all_comb.txt", root + "ent/dev.txt"]

elif args.dev_v2:
    fnames_CCG = [root + "ent/all_comb_rels_v2.txt", root + "ent/dev_rels_v2.txt"]
    fnames_oie = [root + "ent/all_rels_oie.txt", root + "ent/dev_rels_oie.txt"]
    orig_fnames = [root + "ent/all_comb.txt", root + "ent/dev.txt"]

elif args.dev_v3:
    fnames_CCG = [root + "ent/all_comb_rels_v3.txt", root + "ent/dev_rels_v3.txt"]
    fnames_oie = [root + "ent/all_rels_oie.txt", root + "ent/dev_rels_oie.txt"]
    orig_fnames = [root + "ent/all_comb.txt", root + "ent/dev.txt"]

elif args.berDS:
    fnames_CCG = [root + "ent/all_comb_rels.txt", root + "ent/ber_all_rels.txt"]
    orig_fnames = [root + "ent/all_comb.txt", root + "ent/ber_all.txt"]

elif args.berDS_v2:
    fnames_CCG = [root + "ent/all_comb_rels_v2.txt", root + "ent/ber_all_rels_v2.txt"]
    orig_fnames = [root + "ent/all_comb.txt", root + "ent/ber_all.txt"]

elif args.berDS_v3:
    fnames_CCG = [root + "ent/all_comb_rels_v3.txt", root + "ent/ber_all_rels_v3.txt"]
    orig_fnames = [root + "ent/all_comb.txt", root + "ent/ber_all.txt"]

elif args.dev_sherliic_v2:
    fnames_CCG = [root + "ent/all_comb_rels.txt", root + "../sherliic/dev_rels.txt"]
    orig_fnames = [root + "ent/all_comb.txt", root + "../sherliic/sherliic_dev.txt"]

elif args.test_sherliic_v2:
    fnames_CCG = [root + "ent/all_comb_rels.txt", root + "../sherliic/test_rels.txt"]
    orig_fnames = [root + "ent/all_comb.txt", root + "../sherliic/sherliic_test.txt"]

elif args.snli:
    # fnames_CCG = [root + "ent/msnli_rels2.txt", root + "ent/msnli_rels2.txt"]
    # fnames_unary = [root + "ent/msnli_rels_unary2.txt", root + "ent/msnli_rels_unary2.txt"]
    # fnames_CCG = [root + "newsqa/newsqa_test_rels.txt", root + "newsqa/newsqa_test_rels.txt"]
    # fnames_unary = [root + "newsqa/newsqa_test_rels_unary.txt", root + "newsqa/newsqa_test_rels_unary.txt"]
    fnames_CCG = [root + "newsqa/squad_rels.txt", root + "newsqa/squad_rels.txt"]
    fnames_unary = [root + "newsqa/squad_rels_unary.txt", root + "newsqa/squad_rels_unary.txt"]
    orig_fnames = [None, None]

# elif args.calcSupScores or args.supervised:
#     fnames_CCG = [root + "ent/dev_rels.txt", root + "ent/train0_rels.txt"]
#     orig_fnames = [root + "ent/dev.txt", root + "ent/train0.txt"]

elif args.test:
    if args.LDA:
        fnames_CCG = [root+"ent/train_new_LDA15rels.txt",root+"ent/devTrain_new_LDA15rels.txt"]
    else:
        if args.origLevy:
            fnames_CCG = [root + "ent/all_comb_rels.txt", root + "ent/trainTest1_rels.txt"]
            fnames_oie = [root + "ent/all_rels_oie.txt", root + "ent/trainTest1_rels_oie.txt"]
            orig_fnames = [root + "ent/all_comb.txt", root + "ent/trainTest1.txt"]
        else:
            fnames_CCG = [root + "ent/all_comb_rels.txt", root + "ent/test_rels.txt"]
            fnames_oie = [root + "ent/all_rels_oie.txt", root + "ent/test_rels_oie.txt"]
            orig_fnames = [root + "ent/all_comb.txt", root + "ent/test.txt"]

elif args.testok:
    fnames_CCG = ["../../entgraph_gen/ok_rels.txt", root + "ent/test_rels.txt"]
    fnames_oie = [root + "ent/all_rels_oie.txt", root + "ent/test_rels_oie.txt"]
    orig_fnames = [False, root + "ent/test.txt"]

elif args.berDSok:
    fnames_CCG = ["../../entgraph_gen/ok_rels.txt", root + "ent/ber_all_rels.txt"]
    orig_fnames = [False, root + "ent/ber_all.txt"]

elif args.dev_reanno:
    fnames_CCG = [root + "ent/all_comb_rels.txt", root + "ent/dev_rels_reanno.txt"]
    fnames_oie = [root + "ent/all_rels_oie.txt", root + "ent/dev_rels_oie.txt"]
    orig_fnames = [root + "ent/all_comb.txt", root + "ent/dev_reanno.txt"]

elif args.test_reanno:
    fnames_CCG = [root + "ent/all_comb_rels.txt", root + "ent/test_rels_reanno.txt"]
    fnames_oie = [root + "ent/all_rels_oie.txt", root + "ent/test_rels_oie.txt"]
    orig_fnames = [root + "ent/all_comb.txt", root + "ent/test_reanno.txt"]

elif args.test_v2:
    fnames_CCG = [root + "ent/all_comb_rels_v2.txt", root + "ent/test_rels_v2.txt"]
    fnames_oie = [root + "ent/all_rels_oie.txt", root + "ent/test_rels_oie.txt"]
    orig_fnames = [root + "ent/all_comb.txt", root + "ent/test.txt"]

elif args.test_v3:
    fnames_CCG = [root + "ent/all_comb_rels_v3.txt", root + "ent/test_rels_v3.txt"]
    fnames_oie = [root + "ent/all_rels_oie.txt", root + "ent/test_rels_oie.txt"]
    orig_fnames = [root + "ent/all_comb.txt", root + "ent/test.txt"]

# elif args.test_dir:
#     fnames_CCG = [root + "ent/all_comb_rels.txt", root + "ent/test_dir_rels_v2.txt"]
#     fnames_oie = None
#     orig_fnames = [root + "ent/all_comb.txt", root + "ent/test_dir.txt"]

# elif args.dev_dir:
#     fnames_CCG = [root + "ent/all_comb_rels.txt", root + "ent/dev_dir_rels_v2.txt"]
#     fnames_oie = None
#     orig_fnames = [root + "ent/all_comb.txt", root + "ent/dev_dir.txt"]

elif args.test_dir:
    fnames_CCG = [root + "ent/all_comb_rels.txt", root + "ent/official_test_dire/test_dir_rels.txt"]
    fnames_oie = None
    orig_fnames = [root + "ent/all_comb.txt", root + "ent/official_test_dire/test_dir.txt"]

elif args.dev_dir:
    fnames_CCG = [root + "ent/all_comb_rels.txt", root + "ent/official_dev_dire/dev_dir_rels.txt"]
    fnames_oie = None
    orig_fnames = [root + "ent/all_comb.txt", root + "ent/official_dev_dire/dev_dir.txt"]

elif args.strictdire_test:
    fnames_CCG = [root + "ent/all_comb_rels.txt", root + "ent/strictdire_test_rels.txt"]
    fnames_oie = [root + "ent/all_rels_oie.txt", root + "ent/strictdire_test_rels_oie.txt"]
    orig_fnames = [root + "ent/all_comb.txt", root + "ent/strictdire_test.txt"]

elif args.zeichner:
    fnames_CCG = [root + "ent/all_comb_rels.txt", root + "ent/zeichner_rels.txt"]
    fnames_oie = None
    orig_fnames = [root + "ent/all_comb.txt", root + "ent/zeichner.txt"]

elif args.test_naacl:
    fnames_CCG = [root + "ent/all_tensed_rels.txt", root + "ent/naacl_levy_format_rels.txt"]
    fnames_oie = None
    orig_fnames = [root + "ent/all_tensed.txt", root + "ent/naacl_levy_format.txt"]

elif args.test_naacl_untensed:
    fnames_CCG = [root + "ent/naacl_levy_format_untensed_rels.txt", root + "ent/naacl_levy_format_untensed_rels.txt"]
    fnames_oie = None
    orig_fnames = [root + "ent/naacl_levy_format.txt", root + "ent/naacl_levy_format.txt"]
else:
    if args.LDA:
        fnames_CCG = [root+"ent/all_new_LDA15rels.txt",root+"ent/all_new_LDA15rels.txt"]
    else:
        if args.dsPath:
            fnames_CCG = [root + args.dsPath +"_rels.txt", root + args.dsPath +"_rels.txt"]
        else:
            fnames_CCG = [root+"ent/all_rels.txt",root+"ent/all_rels.txt"]

    fnames_oie = [root+"ent/all_rels_oie.txt",root+"ent/all_rels_oie.txt"]

    if args.dsPath:
        orig_fnames = [root + args.dsPath +".txt", root + args.dsPath +".txt"]
    else:
        orig_fnames = [root + "ent/all.txt", root + "ent/all.txt"]

#parameters

CCG = args.CCG
typed = args.typed
supervised = args.supervised
oneFeat = args.oneFeat#as opposed to average features!
gpath = args.gpath
method = args.method
useSims = args.useSims
if gpath:
    print ("gpath: ", gpath)
if method is None:
    method = "x"
write = args.write

if debug:
    print ("args.tnf: ", args.tnf)

if args.tnf:
    from graph import graph_holder
    read_sims = False
elif args.instance_level:
    read_sims = False
else:
    read_sims = True

if debug:
    print ("args: ", CCG)

if gpath:
    engG_dir_addr = "../../gfiles/" + gpath +"/"
    if debug:
        print ("dir_addr: ", engG_dir_addr)
else:
    engG_dir_addr = None

if CCG:
    fnames = fnames_CCG
    fname_feats = None
    sim_path = root + "ent/ccg.sim"
    if not gpath:
        if not args.featsFile:
            raise Exception("featsFile not provided")
        else:
            fname_feats = root + "ent/" + args.featsFile + ".txt"
else:
    fnames = fnames_oie
    fname_feats = None
    sim_path = root + "ent/oie.sim"
    if not gpath:
        if not args.featsFile:
            raise Exception("featsFile not provided")
        else:
            fname_feats = root + "ent/" + args.featsFile + ".txt"


rels2Sims = evaluation.util.read_rels_sim(sim_path, CCG, useSims)

#end parameters


#Form the samples (dev will contain the test if you use --test instead of --dev
[_, Y_dev_base] = [baseline.predict_lemma_baseline(fname, args) for fname in orig_fnames]

#Do the training and prediction!

if orig_fnames[0]:
    lines_dev = open(orig_fnames[1]).read().splitlines()
else:
    lines_dev = None


if not args.instance_level:
    data_list, predPairSumCoefs, predPairFeats, predPairFeatsTyped, predPairConnectedList, \
    predPairConnectedWeightList, predPairTypedConnectedList, predPairTypedConnectedWeightList, predPairTypedExactFound = form_samples(fnames,
                                                                                                           fnames_unary,
                                                                                                           orig_fnames,
                                                                                                           engG_dir_addr,
                                                                                                           fname_feats,
                                                                                                           rels2Sims,
                                                                                                           args)
    if not args.snli:
        Y_dev_pred0, Y_dev_TNF0, Y_dev_TNF_typed0, cl, scaler = fit_predict(data_list, predPairFeats,
                                                                            predPairFeatsTyped, predPairConnectedList,
                                                                            predPairTypedConnectedList,
                                                                            predPairTypedExactFound,
                                                                            args)
elif args.instance_level:
    data_list = [evaluation.util.read_data(fnames[i], orig_fnames[i], args.CCG, args.typed, args.LDA) for i in
                 range(len(fnames))]
    Y_dev_pred0 = evaluation.util.read_instance_level_probs(fname_feats)
    if not args.no_lemma_baseline:
        assert len(Y_dev_pred0) == len(Y_dev_base)

    Y_dev_TNF0, Y_dev_TNF_typed0, cl, scaler, predPairSumCoefs, predPairFeats, predPairFeatsTyped = None, None, None, None, None, None, None


#only calc the supervised scores and quit
if args.calcSupScores:

    read_sims = True
    assert args.supervised
    assert not args.useSims
    assert not args.write
    assert args.supervised
    assert not args.oneFeat
    assert not args.useSims


    files = os.listdir(engG_dir_addr)
    files = list(np.sort(files))
    num_f = 0
    for f in files:

        if num_f % 50 == 0:
            print ("num processed files: ", num_f)

        gpath=engG_dir_addr+f
        # if f not in acceptableFiles:
        #     continue

        if f_post_fix not in f or os.stat(gpath).st_size == 0:
            continue
        print ("fname: ", f)
        num_f += 1

        gr = graph.Graph(gpath=gpath, args = args)
        gr.set_Ws()

        if num_feats==-1 and read_sims:
            num_feats = 2*gr.num_feats

        calcGraphScores(gr,cl,scaler)
        del gr

#do the final evaluation (either raw scores or graphs)
elif not args.snli:

    data_dev = data_list[1]
    data_dev_CCG = evaluation.util.read_data(fnames_CCG[1], orig_fnames[1], args.CCG, args.typed, args.LDA)
    Y_dev = [l for (_,_,_,_,_,_,l) in data_dev]

    print ("baseline eval:")
    eval(Y_dev_base,Y_dev)

    if debug:
        print (Y_dev_pred0)

    _, Y_berant = final_prediction(data_dev,data_dev_CCG, predPairFeats ,predPairFeatsTyped, predPairSumCoefs, Y_dev_base, Y_dev_pred0,Y_dev_TNF0,Y_dev_TNF_typed0, lines_dev, root, args)

    if debug:
        print ("Y_dev_TNF0"), Y_dev_TNF0

    if Y_dev_TNF0 is not None:

        s1 = root + out_dir
        if not os.path.isdir(s1):
            os.mkdir(s1)

        op_pr_rec_TNF = open(s1 + args.method + "_TNF.txt",'w')
        op_pr_rec_TNF_typed = open(s1 + args.method + "_TNF_typed.txt",'w')

        if debug:
            print ("pr rec TNF: ")
        for i in range(len(Y_dev_TNF0)):
            pr, rec, _ = eval(Y_dev_TNF0[i],Y_dev)
            op_pr_rec_TNF.write(str(pr)+"\t"+str(rec)+"\n")
            if debug:
                print (pr, " ", rec)

        if debug:
            print ("pr rec TNF Typed: ")
        for i in range(len(Y_dev_TNF_typed0)):

            if debug:
                print ("typed ", i)

            Y_dev_pred_binary = Y_dev_TNF_typed0[i]
            if debug:
                write_detailed_res_binary(data_dev,lines_dev,Y_dev,Y_dev_pred_binary,Y_dev_pred_binary, Y_berant)

            pr, rec, _ = eval(Y_dev_TNF_typed0[i],Y_dev)
            op_pr_rec_TNF_typed.write(str(pr)+"\t"+str(rec)+"\n")
            if debug:
                print (pr, " ", rec)
