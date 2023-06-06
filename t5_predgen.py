import os
import time
import torch
import nltk
from itertools import chain

from transformers import BertModel, BertTokenizer, BertForPreTraining, T5Tokenizer, T5ForConditionalGeneration
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.stem import WordNetLemmatizer
from pattern.en import conjugate, lemma, lexeme, PRESENT, SG, PL, PAST, PROGRESSIVE
from copy import deepcopy
import argparse
import re

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--beam', type=int,
                        help="beam expansion size for every p in one layer",
                        default=100)
    parser.add_argument("--keep", type=int,
                        help="at least how much p per graph",
                        default=1000)
    parser.add_argument("--filter_level", type=int,
                        help="the level for filtering; 0: no filtering; 1: typestring and 'and' shouldn't appear",
                        default=0)
    parser.add_argument("--t5_size", type=str,
                        help="",
                        default='small')
    parser.add_argument("--model_path", type=str,
                        help="",
                        default='NONE')
    args = parser.parse_args()
    print(args)
    print(os.getpid())

    BEAM = args.beam
    KEEP = args.keep
    preds_dir = 'generated_preds_'+str(BEAM)+'_multihop_'+str(KEEP)+'_fil'+str(args.filter_level)+'_2nei_'+args.t5_size+'_tuned'+'/'
    if args.model_path != 'NONE':
        preds_dir = 'generated_preds_'+str(BEAM)+'_multihop_'+str(KEEP)+'_fil'+str(args.filter_level)+'_2nei_'+args.t5_size+'_tuned'+''.join(args.model_path[:-4].split('_')[2:])+'/'
    preds_dir = preds_dir[:-1]+'_reannoseed/'
    # preds_dir = preds_dir.replace('_2nei', '')
    if not os.path.exists(preds_dir):
        os.mkdir(preds_dir)
    print('preds dir', preds_dir)

model = BertForPreTraining.from_pretrained('bert-large-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
wnl = WordNetLemmatizer()
t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
if __name__ == '__main__':
    t5_tokenizer = T5Tokenizer.from_pretrained('t5-'+args.t5_size)
    t5_model = T5ForConditionalGeneration.from_pretrained('t5-'+args.t5_size)
    if args.model_path != 'NONE':
        t5_model.load_state_dict(torch.load(args.model_path))

t5_model.cuda()

gg_occured = {}

try:
    conjugate(verb='gg',tense=PRESENT,aspect=PROGRESSIVE)
except:
    pass

with open('levy_types.txt','r') as f_type:
    types = list(f_type.readlines())
types = [t.strip().split('#') for t in types]

type2template = {}
type2templatelen = {}
wordtags = nltk.ConditionalFreqDist((w.lower(), t) for w, t in chain(nltk.corpus.brown.tagged_words(tagset="universal"), nltk.corpus.treebank.tagged_words(tagset="universal"))) 
with open('type_map.txt', 'r') as f:
    for l in f.readlines():
        ll = l.strip().split('\t')
        type2template[ll[0]] = ll[1]
        type2templatelen[ll[0]] = len(ll[1].split())


prep_list = ['in', 'on', 'with', 'by', 'for', 'at', 'about', 'under', 'of', 'into', 'within', 'throughout', 'inside', 'outside', 'without', 'to']

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

def is_verb(token):
    return 'VERB' in wordtags[token]

def is_ving(token):
    if len(wordtags[token]) == 0:
        return token[-3:] == 'ing'
    if not is_verb(token):
        return False
    return token == conjugate(verb=token,tense=PRESENT,aspect=PROGRESSIVE)

def is_ved(token):
    if len(wordtags[token]) == 0:
        return token[-2:] == 'ed'
    if not is_verb(token):
        return False
    return token == conjugate(verb=token,tense=PAST,aspect=PROGRESSIVE)

def is_adj(token):
    if 'VERB' not in wordtags[token]:
        return 'ADJ' in wordtags[token]
    return 'ADJ' in wordtags[token] and wordtags[token]['ADJ'] > wordtags[token]['VERB']

def is_adv(token):
    # if 'VERB' not in wordtags[token]:
    #     return 'ADV' in wordtags[token]
    return 'ADV' in wordtags[token] and wordtags[token]['ADV'] > wordtags[token]['ADJ']


def is_noun(token):
    if 'VERB' not in wordtags[token]:
        return 'NOUN' in wordtags[token]
    return 'NOUN' in wordtags[token] and wordtags[token]['NOUN'] > wordtags[token]['VERB']

def is_prep(token):
    return token in prep_list or 'ADP' in wordtags[token]

def t5_generation(input_sent, t5_tokenizer, t5_model, num_return_sequences, num_beams, max_length=7):
    input_ids = t5_tokenizer(input_sent, return_tensors="pt").input_ids
    outputs = t5_model.generate(input_ids.cuda(), max_length=max_length, num_return_sequences=num_return_sequences, num_beams=num_beams)
    outputs = t5_tokenizer.batch_decode(outputs)
    outputs = [s.split('<extra_id_0>')[1].split('<extra_id_1>')[0].strip() for s in outputs if '<extra_id_0>' in s]
    return outputs

token_should_be_filtered = ['not', 'does', 'doesn\'t', 'doesn’t', 'didn\'t', 'NOT', 'also', 'actually', 'currently', 'really', 'particularly', 'can', 'cannot', 'could', 'cann\'t', 'wouldn\'t', 'cann’t', 'won\'t', 'wouldn’t', 'will', 'would', 'should', 'shall', 'that', 'especially', 'often', 'usually', 'now', 'still', 'may', 'already', 'generally', 'a', 'an', 'must', 'haven\'t', 'hasn\'t']
negative_tokens = set(['not', 'NOT', 'don\'t', 'doesn\'t', 'doesn’t', 'didn\'t', 'cannot', 'cann\'t', 'won\'t', 'wouldn\'t', 'cann’t', 'wouldn’t', 'haven\'t', 'hasn\'t'])

def t5_generated2pred(inputs):
    preds = set()
    for ss in inputs:
        # head should be verb
        # tail should be verb or prep
        s = ss.lower().split()
        prefix = '('
        if len(s) == 0:
            continue
        if s[0] == 'isn\'t' or s[0] == 'aren\'t':
            s[0] = 'be'
            prefix = 'NEG__('
        if len(set(s).intersection(negative_tokens)) > 0:
            prefix = 'NEG__('
        s = list(filter(lambda x: x not in token_should_be_filtered, s))
        if s[:2] == ['have', 'been'] or s[:2] == ['has', 'been']:
            s = s[1:]
        if len(s)>1 and s[0] in ['have', 'has', 'had'] and is_ved(s[1]):
            s = s[1:]
        if len(s)>2 and s[0] in ['have', 'has', 'had'] and s[1]=='to':
            s = s[2:]
        if len(s) == 0:
            continue
        head_idx = 0
        tail_idx = len(s)-1
        if 'b' in s:
            tail_idx = s.index('b')
        while head_idx<=tail_idx and head_idx<len(s) and not is_verb(s[head_idx]):
            head_idx += 1
        while head_idx<=tail_idx and tail_idx>=0 and not (is_verb(s[tail_idx]) or is_prep(s[tail_idx])):
            tail_idx -= 1
        if head_idx > tail_idx:
            continue
        pruned_s = s[head_idx:tail_idx+1]
        head_lmtz = wnl.lemmatize(pruned_s[0], 'v')
        if head_lmtz == 'be':
            if len(pruned_s) > 1 and is_ving(pruned_s[1]):
                pruned_s = pruned_s[1:]
        head_lmtz = wnl.lemmatize(pruned_s[0], 'v')
        if head_lmtz == 'be' and not (len(pruned_s)>1 and is_prep(pruned_s[1])):
            if len(pruned_s) == 1:
                preds.add(prefix+'be.1,be.2)')
                continue
            if is_adv(pruned_s[1]) and len(pruned_s)>2:
                pruned_s = pruned_s[0:1]+pruned_s[2:]
            if is_adj(pruned_s[1]) and is_prep(pruned_s[-1]):
                pruned_s[1] = wnl.lemmatize(pruned_s[1], 'a')
                preds.add(prefix+pruned_s[1]+'.1,'+'.'.join(pruned_s[1:])+'.2)')
                continue
            if is_noun(pruned_s[1]) and is_prep(pruned_s[-1]):
                pruned_s[1] = wnl.lemmatize(pruned_s[1], 'n')
                preds.add(prefix+pruned_s[1]+'.1,'+'.'.join(pruned_s[1:])+'.2)')
                continue
            if is_ved(pruned_s[1]):
                if is_prep(pruned_s[-1]):
                    pruned_s[1] = wnl.lemmatize(pruned_s[1], 'v')
                    preds.add(prefix+pruned_s[1]+'.2,'+'.'.join(pruned_s[1:])+'.2)')
                else:
                    pruned_s[1] = wnl.lemmatize(pruned_s[1], 'v')
                    preds.add(prefix+pruned_s[1]+'.2,'+'.'.join(pruned_s[1:])+'.3)')
                    preds.add(prefix+pruned_s[1]+'.1,'+'.'.join(pruned_s[1:])+'.3)')
                continue
            if ss not in gg_occured:
                gg_occured[ss] = 1
                print("t5_generated2pred UNEXPECTED CASE A!", s, pruned_s, ss)
        # Now the head should be verb but not 'be' !
        if len(pruned_s) == 1:
            pruned_s[0] = wnl.lemmatize(pruned_s[0], 'v')
            preds.add(prefix+pruned_s[0]+'.1,'+pruned_s[0]+'.2)')
        elif is_prep(pruned_s[-1]):
            pruned_s[0] = wnl.lemmatize(pruned_s[0], 'v')
            preds.add(prefix+pruned_s[0]+'.1,'+'.'.join(pruned_s)+'.2)')
        else:
            if ss not in gg_occured:
                gg_occured[ss] = 1
                print("t5_generated2pred UNEXPECTED CASE B!", s, pruned_s, ss)
    return preds

def f_level1(pred, word):
    p = pred.split('.')
    return all(map(lambda token: token not in word and token[:-1] not in word, p))

conjs = [', which entails that ', ', that is to say, ', ' and therefore ']

def seed2neighbors(seed_pred, t5_tokenizer, t5_model, num_return_sequences, num_beams, max_length=7, filter_level=0, conj=0):
    typeList = sorted(seed_pred.split('#')[1:])
    typeList2 = deepcopy(typeList)
    if typeList[0][-2] == '_':
        typeList2[0] = typeList2[0][:-2]
        typeList2[1] = typeList2[1][:-2]
    sent = pred2template(seed_pred, typeList2)

    sent1 = sent + conjs[conj] + type2template[typeList2[0]] + 'A <extra_id_0> ' + type2template[typeList2[1]] + 'B.'
    sent2 = sent + conjs[conj] + type2template[typeList2[1]] + 'B <extra_id_0> ' + type2template[typeList2[0]] + 'A.'
    generated_phrases = t5_generation(sent1, t5_tokenizer, t5_model, num_return_sequences // 2, num_beams, max_length=max_length)
    preds = t5_generated2pred(generated_phrases)
    preds = [p.replace('#', '') + '#' + typeList[0] + '#' + typeList[1] for p in preds if p.count(',') == 1]

    generated_phrases = t5_generation(sent2, t5_tokenizer, t5_model, num_return_sequences // 2, num_beams, max_length=max_length)
    preds2 = t5_generated2pred(generated_phrases)
    preds2 = [p.replace('#', '') + '#' + typeList[1] + '#' + typeList[0] for p in preds2 if p.count(',') == 1]

    ret_set = set(preds) | set(preds2)

    if filter_level == 0:
        return ret_set
    if filter_level > 0:
        typestrings = set(typeList2[0].lower().split('_')+typeList2[1].lower().split('_')+['and'])
        ret_set = set(
            filter(
                lambda p: all(
                    map(
                        lambda token: token not in typestrings and token[:-1] not in typestrings, 
                        re.split(r'[\.\,]', p.split('#')[0][1:-1])
                    )
                )
            , ret_set)
            )
    return ret_set


def multihop_gen(seeds_with_same_type, size, t5_tokenizer, t5_model, num_return_sequences, num_beams, max_length=7, filter_level=0):
    generated = set(seeds_with_same_type)
    cur_layer = set(seeds_with_same_type)
    while len(generated) < size:
        next_layer = set()
        for p in cur_layer:
            next_layer = next_layer | seed2neighbors(p, t5_tokenizer, t5_model, num_return_sequences, num_beams, max_length, filter_level)
        next_layer = next_layer - generated
        cur_layer = next_layer
        generated = generated | next_layer
        if len(cur_layer) == 0:
            print(list(seeds_with_same_type)[0].split('#')[1:], 'generation converge! size =', len(generated))
            break
    return generated
    
def multihop_gen_2nei(seeds_with_same_type, size, t5_tokenizer, t5_model, num_return_sequences, num_beams, max_length=7, filter_level=0):
    generated = set(seeds_with_same_type)
    cur_layer = set(seeds_with_same_type)
    occured_once = set()
    while len(generated) < size:
        next_layer = set()
        for p in cur_layer:
            new_generated = seed2neighbors(p, t5_tokenizer, t5_model, num_return_sequences, num_beams, max_length, filter_level)
            new_generated = new_generated - generated
            next_layer = next_layer | (new_generated & occured_once)
            occured_once = occured_once ^ new_generated

        next_layer = next_layer - generated
        cur_layer = next_layer
        generated = generated | next_layer
        if len(cur_layer) == 0:
            print(list(seeds_with_same_type)[0].split('#')[1:], 'generation converge! size =', len(generated))
            break
    return generated

seeds = []
gr = {}

print('--test--')
print(seed2neighbors('(different.1,different.from.2)#location_2#location_1', t5_tokenizer, t5_model, 20, 10, max_length=7))
print('--test--')

if __name__ == '__main__':
    pos_pred_set = set()
    with open('../gfiles/ent/dev_rels_reanno.txt', 'r') as f:
    # with open('../sherliic/dev_rels.txt', 'r') as f:
        for l in f.readlines():
            ll = l.strip().split('\t')
            if len(ll) < 3 or ll[2] == 'False':
                continue
            if len(ll[0])>0:
                pos_pred_set.add(ll[0])
            if len(ll[1])>0:
                pos_pred_set.add(ll[1])
    for pred in pos_pred_set:
        p, pt1, pt2 = pred.split(' ')
        pt1 = pt1.split('::')[1]
        pt2 = pt2.split('::')[1]
        p = p + '#' + pt1 + '#' + pt2
        seeds.append(p)
        typepair = '#'.join(sorted([pt1,pt2]))
        if typepair not in gr:
            gr[typepair] = set()
        if pt1 == pt2:
            p = pred.split(' ')[0] + '#' + pt1 + '_1' + '#' + pt2 + '_2'
        gr[typepair].add(p)
    print(len(seeds))

    for t in sorted(gr.keys()):
        generated = multihop_gen_2nei(gr[t], KEEP, t5_tokenizer, t5_model, num_return_sequences=BEAM, num_beams=BEAM, max_length=7, filter_level=args.filter_level)
        gr[t].update(generated)
        print('=====')
        print(t, len(gr[t]))
        print(gr[t])

        t1, t2 = t.split('#')
        with open(preds_dir+t1+'#'+t2, 'w') as f:
            for i, p in enumerate(gr[t]):
                f.write('\t'.join(p.split('#'))+'\n')