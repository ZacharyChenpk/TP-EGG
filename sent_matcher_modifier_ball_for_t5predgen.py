from re import template
import sys
import numpy as np
sys.path.append("..")
from copy import deepcopy
import time
from itertools import chain
import os
from multiprocessing.dummy import Pool
from functools import reduce

import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from bert_nli import BertNLIModel
from transformers import BertModel, BertTokenizer
from pattern.en import conjugate, lemma, lexeme, PRESENT, SG, PL, PAST, PROGRESSIVE
import torch
import my_graph as graph
from args import args
import utils.utils as utils

torch.cuda.set_device(3)
print('device:',3,os.getpid())
N_PART = 4
PART_NO = -1

LEMMATIZER = WordNetLemmatizer()
STOPWORDS = stopwords.words('english')

REMAIN_TYPE = 2
# FACTOR = 5e6
FACTOR = int(2e7)
FACTOR_STR = '2e7'
MINSCORE = None
BEAM = 50
KEEP = 5000
FIL = 0
SUFFIX = '_large_tunedlarge0.001reannofix'
dim = 4
dfunc = 'exp'
embmod_dim = 16

# print "stopwords: ", STOPWORDS
# We remove 'own' and a few other form stopwords as in our domain it's not really a stop word
STOPWORDS.remove('own')
STOPWORDS.remove('does')
STOPWORDS.remove('do')
STOPWORDS.remove('doing')

def get_lemmas(phrase, pos=wn.VERB):
    return [LEMMATIZER.lemmatize(w, pos) for w in phrase.split(' ')]

def get_lemmas_no_stopwords(phrase, pos=wn.VERB):
    return set([w for w in get_lemmas(phrase, pos) if w not in STOPWORDS])

def aligned_args(q, a):
    q_arg = get_lemmas_no_stopwords(q[2], wn.NOUN)
    if q_arg == get_lemmas_no_stopwords(a[2], wn.NOUN):
        return True
    if q_arg == get_lemmas_no_stopwords(a[0], wn.NOUN):
        return False
    return -1

def aligned_args_rel(q, a):
    # These are not necessary if the sentences are well formed!
    q1 = LEMMATIZER.lemmatize(q[1].split("::")[0].lower())
    q2 = LEMMATIZER.lemmatize(q[2].split("::")[0].lower())
    a1 = LEMMATIZER.lemmatize(a[1].split("::")[0].lower())
    a2 = LEMMATIZER.lemmatize(a[2].split("::")[0].lower())

    if q1 == a1:
        return True
    elif q1 == a2:
        return False
    else:
        if q2 == a1:
            return False
        elif q2 == a2:
            return True
        print ("not sure if aligned: ", q, a)
        return True  # This is a bad case!

def read_data(dpath, orig_dpath,CCG,typed,LDA):
    f = open(dpath)

    if orig_dpath:
        lines_orig = open(orig_dpath).read().splitlines()
    else:
        lines_orig = None

    data = []
    idx = 0
    for l in f:
        line = l.replace("\n","")

        # if idx==10000:
        #     break

        if lines_orig:
            line_orig = lines_orig[idx]
        else:
            line_orig = None

        ss = line.split("\t")

        if len(ss)<3:
            print ("bad len problem: ", line)
            idx += 1
            continue

        q_all = ss[0].split(" ")
        p_all = ss[1].split(" ")
        q = q_all[0]
        p = p_all[0]

        if len(p_all)>1 and typed and not LDA:
            try:
                t1 = p_all[1].split("::")[1]
                t2 = p_all[2].split("::")[1]
            except:
                t1 = "thing"
                t2 = "thing"
            t1s = [t1]
            t2s = [t2]
            probs = [1]
        elif typed and LDA:
            tss = ss[3].split()
            i=0
            t1s = []
            t2s = []
            probs = []
            while i<len(tss):
                ts = tss[i].split("#")
                t1 = ts[0]
                t2 = ts[1]

                i+=1
                prob = float(tss[i])
                t1s.append(t1)
                t2s.append(t2)
                probs.append(prob)
                i+=1
        else:
            #Not well formed
            t1 = "thing"
            t2 = "thing"
            t1s = [t1]
            t2s = [t2]
            probs = [1]

        #First, let's see if the args are aligned

        if CCG:
            a = True
            if line_orig:#lazy way to check snli
                if len(q_all)>1 and len(p_all)>1:
                    a = aligned_args_rel(q_all,p_all)

        else:
            if line_orig:
                ss_orig = line_orig.split("\t")
                q_orig = ss_orig[0].split(",")
                p_orig = ss_orig[1].split(",")

                a = aligned_args([q_orig[0].strip(),"_",q_orig[2].strip()],[p_orig[0].strip(),"",p_orig[2].strip()])
                if a==-1:
                    a = aligned_args([p_orig[0].strip(),"",p_orig[2].strip()],[q_orig[0].strip(),"_",q_orig[2].strip()])
                    if a==-1:
                        raise Exception('HORRIBLE BUG!!!'+str(q)+" "+str(a))
            else:
                a = True

        try:
            q_arg1 = LEMMATIZER.lemmatize(q_all[1].split("::")[0])
            q_arg2 = LEMMATIZER.lemmatize(q_all[2].split("::")[0])

            p_arg1 = LEMMATIZER.lemmatize(p_all[1].split("::")[0])
            p_arg2 = LEMMATIZER.lemmatize(p_all[2].split("::")[0])

        except:
            print ("problem: ", line)
        #(exports.1,exports.2) nigeria oil	(supplier.of.1,supplier.of.2) nigeria oil

        if ss[2].startswith("n") or ss[2]=="False":
            l = 0
        else:
            l = 1

        data.append((p,q,t1s,t2s,probs,a,l))
        idx += 1

    return data

def min_match_length(A, B):
    for i in range(min(len(A), len(B))):
        if A[i]!=B[i]:
            return i
    return min(len(A),len(B))

type2template = {}
wordtags = nltk.ConditionalFreqDist((w.lower(), t) for w, t in chain(nltk.corpus.brown.tagged_words(tagset="universal"), nltk.corpus.treebank.tagged_words(tagset="universal"))) 
with open('type_map.txt', 'r') as f:
    for l in f.readlines():
        ll = l.strip().split('\t')
        type2template[ll[0]] = ll[1]

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

root = "../gfiles/"
model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('/home/chenzhb/bert-base-uncased-vocab.txt')
d_net = torch.nn.Sequential(
    torch.nn.Linear(768, dim),
    torch.nn.ReLU(),
    torch.nn.Linear(dim, 1)
)


checkpoint_file = 'sent_matchers/ball2_alltest_bertbase_0.9_1e-05_0.0005_5_'+dfunc+'_'+str(dim)+'_best.pth.tar'
if embmod_dim is not None:
    checkpoint_file = checkpoint_file[:-13]+'_em'+str(embmod_dim)+'_best.pth.tar'
checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
print('loading from', checkpoint_file)
model.load_state_dict(checkpoint['state_dict'])
model = model.cuda()
d_net.load_state_dict(checkpoint['d_net'])
d_net.cuda()
if embmod_dim is not None:
    embmod_net = torch.nn.Sequential(
        torch.nn.Linear(768, embmod_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(embmod_dim, embmod_dim)
    )
    embmod_net.load_state_dict(checkpoint['embmod_net'])
    embmod_net.cuda()

nli = BertNLIModel(model_path=None,gpu=torch.cuda.is_available(),bert_type='deberta',label_num=3,batch_size=1024,reinit_num=-1,freeze_layers=True)

checkpoint_file = 'deberta_tars/deberta0.8_12_1e-05_1_reannofix_best.pth.tar'
checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
# checkpoint_file = 'NO deberta ckpt'
print('loading from', checkpoint_file)
nli.load_state_dict(checkpoint['state_dict'])
nli = nli.cuda()

engG_dir_addr = "generated_preds_"+str(BEAM)+"_multihop_"+str(KEEP)+'_fil'+str(FIL)+'_2nei'+SUFFIX
write_addr = "../gfiles/typedEntGrDir_sent_matcher_t5gen_b"+str(BEAM)+"k"+str(KEEP)+"fil"+str(FIL)+"2nei"+SUFFIX+"_ball2m_1e-5_5e-4_5_"+dfunc+"_"+str(dim)+"_f"+FACTOR_STR+'_m'+str(MINSCORE)+"_ty"+str(REMAIN_TYPE)+"/"
if embmod_dim is not None:
    write_addr = "../gfiles/typedEntGrDir_sent_matcher_t5gen_b"+str(BEAM)+"k"+str(KEEP)+"fil"+str(FIL)+"2nei"+SUFFIX+"_ball2m_1e-5_5e-4_5_"+dfunc+"_"+str(dim)+"_em"+str(embmod_dim)+"_f"+FACTOR_STR+'_m'+str(MINSCORE)+"_ty"+str(REMAIN_TYPE)+"/"

if not os.path.exists(write_addr):
    os.mkdir(write_addr)
print('output', write_addr)
args.featIdx = 4 if 'C_3_3' in engG_dir_addr else 0
files = os.listdir(engG_dir_addr)
files = list(np.sort(files))
print('graphs loading from', engG_dir_addr)

if N_PART > 1 and PART_NO >= 0:
    print('N_PART', N_PART, 'PART_NO', PART_NO)
    files_part_size = (len(files) // N_PART) + 1
    files = files[PART_NO*files_part_size:(PART_NO+1)*files_part_size]
    print('current part', files)

file_clean = []
file_dirty = False

for f in files:
    if os.path.exists(write_addr + f + '.txt'):
        file_clean.append(file_dirty)
        file_dirty = f

for f in files:
    if f in file_clean:
        print(f, 'passed!')
        continue
    print(f, 'start')
    stt = time.time()
    gpath=os.path.join(engG_dir_addr, f)

    gr = graph.Graph(gpath=-1, from_raw=False)
    graph.Graph.featIdx = args.featIdx
    gr.pred2idx = {}
    gr.idxpair2score = {}
    gr.allPreds = []

    first = True
    #read the lines and form the graph
    lIdx = 0
    fopen = open(gpath)
    for line in fopen:
        line = line.replace("` ","").rstrip()
        lIdx += 1
        if first:
            gr.name = gpath
            gr.rawtype = f
            gr.types = f.split("#")
            if len(gr.types) == 2:
                if gr.types[0]==gr.types[1]:
                    gr.types[0] += "_1"
                    gr.types[1] += "_2"
            first = False
        pred = '#'.join(line.strip().split('\t'))

        #The node
        if pred not in gr.pred2idx:
            gr.pred2idx[pred] = len(gr.pred2idx)
            gr.allPreds.append(pred)
    fopen.close()
    gr.set_Ws()
    if len(gr.allPreds) > 0:
        with torch.no_grad():
            template_list = [pred2template(p, gr.types) for p in gr.allPreds]
            bsz = 8192
            sent_pl = None
            for batchid in range(0, len(template_list), bsz):
                template_batch = template_list[batchid:batchid+bsz]
                sent_inputs = tokenizer(template_batch, return_tensors="pt", padding=True)
                for key in sent_inputs.keys():
                    sent_inputs[key] = sent_inputs[key].cuda()
                outputs = model(**sent_inputs)
                if sent_pl is None:
                    sent_pl = outputs.pooler_output
                else:
                    sent_pl = torch.cat([sent_pl, outputs.pooler_output], dim=0)
            print('sent_pl get', 'time:', time.time() - stt)

            dpq = d_net(sent_pl).squeeze(1)
            if embmod_dim is not None:
                sent_pl = embmod_net(sent_pl)
            if dfunc == 'exp':
                dpq = dpq.exp()
            elif dfunc == 'sqr':
                dpq = dpq**2
            else:
                raise Exception('Not Implemented')

            if sent_pl.size(0) <= 2000:
                d = (sent_pl.unsqueeze(1)-sent_pl.unsqueeze(0)).norm(p=2,dim=2)
                # match_scores = (dpq.unsqueeze(0)-d)/(2*dpq.unsqueeze(1))
                match_scores = 2*(dpq.unsqueeze(0)-d)/(dpq.unsqueeze(1))
                print('match scores get', 'time:', time.time() - stt)
                if REMAIN_TYPE == 1:
                    _, matched_indices_q = match_scores.topk(k=min(len(template_list), int(FACTOR//len(template_list))),dim=1)
                    matched_indices_p = torch.arange(len(template_list)).cuda().unsqueeze(1).repeat(1,matched_indices_q.size(1))
                    indices = torch.stack([matched_indices_p, matched_indices_q], dim=2).reshape(-1,2)
                elif REMAIN_TYPE == 2:
                    _, matched_indices = match_scores.view(-1).topk(k=min(FACTOR, len(template_list)**2),dim=0)
                    indices_row = matched_indices.div(len(template_list), rounding_mode='trunc')
                    indices_col = matched_indices.remainder(len(template_list))
                    indices = torch.stack([indices_row, indices_col], dim=1)
                    assert indices.size(1) == 2
            else:
                # sent_pl_chunks = sent_pl.split(mscore_bsz)
                if REMAIN_TYPE == 1:
                    raise Exception('Not Implemented')
                assert REMAIN_TYPE == 2
                match_scores = torch.FloatTensor([]).to(sent_pl.device)
                matched_indices = torch.LongTensor([]).to(sent_pl.device)
                checked_num = 0
                mscore_bsz = int(4e6) // sent_pl.size(0)
                sent_pl_chunks = sent_pl.split(mscore_bsz)
                dpq_chunks = dpq.split(mscore_bsz)
                for left_pl, dpq_chunk in zip(sent_pl_chunks, dpq_chunks):
                    # print(left_pl.size(), sent_pl.size())
                    d = [(left_pl.unsqueeze(1)-sent_pl[bid:bid+mscore_bsz].unsqueeze(0)).norm(p=2,dim=2) for bid in range(0, len(template_list), mscore_bsz)]
                    d = torch.cat(d, dim=1)
                    # row_chunk = ((dpq.unsqueeze(0)-d)/(2*dpq_chunk.unsqueeze(1))).view(-1)
                    row_chunk = (2*(dpq.unsqueeze(0)-d)/(dpq_chunk.unsqueeze(1))).view(-1)
                    candidates = torch.cat([match_scores, row_chunk.sigmoid()], dim=0)
                    cand_idx = torch.cat([matched_indices, torch.arange(row_chunk.size(0)).to(sent_pl.device)+checked_num], dim=0)
                    match_scores, matched_cand_idx = candidates.topk(k=min(candidates.size(0), FACTOR), dim=0)
                    matched_indices = cand_idx[matched_cand_idx]
                    checked_num += int(row_chunk.size(0))
                    del d
                    del row_chunk
                indices_row = matched_indices.div(len(template_list), rounding_mode='trunc')
                indices_col = matched_indices.remainder(len(template_list))
                indices = torch.stack([indices_row, indices_col], dim=1).long()
                print(matched_indices.max(), len(template_list))
            
            # print(matched_indices_p, matched_indices_q, indices)
            templatepair_list = [(template_list[i], template_list[j]) for i,j in list(indices)]
            print('templatepair_list build', 'time:', time.time() - stt)

            indices = indices.T
            if 'deberta' in nli.bert_type:
                values = nli(templatepair_list)[1][:,2]
            else:
                values = nli(templatepair_list)[1][:,1]
            print('model forward', 'time:', time.time() - stt)

        utils.sen2index_memo = {}
        utils.sen2token_memo = {}
        size = torch.Size([len(gr.pred2idx), len(gr.pred2idx)])
        gr.composing(indices,values,size)
        gr.writeGraphToFile(write_addr + f + '.txt')
        print(f, 'nnz', values.size(0), 'time:', time.time() - stt)
