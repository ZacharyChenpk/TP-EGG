from multiprocessing.dummy import Pool
n_pool = 0

sen2index_memo = {}
sen2token_memo = {}

def get_pair_input(tokenizer, sent1, sent2, model_type):
    # if 'roberta' in model_type:
    #     text = "<s> {} </s></s> {} </s>".format(sent1, sent2)
    # else:
    #     text = "[CLS] {} [SEP] {} [SEP]".format(sent1,sent2)

    # tokenized_text = tokenizer.tokenize(text)
    # indexed_tokens = tokenizer.encode(text)[1:-1]
    # assert len(tokenized_text) == len(indexed_tokens)
    # =====
    if sent1 in sen2index_memo:
        tokenized_text_1 = sen2token_memo[sent1]
        indexed_tokens_1 = sen2index_memo[sent1]
    else:
        tokenized_text_1 = tokenizer.tokenize(' '+sent1+' ')
        indexed_tokens_1 = tokenizer.encode(' '+sent1+' ')[1:-1]
        sen2token_memo[sent1] = tokenized_text_1
        sen2index_memo[sent1] = indexed_tokens_1
    if sent2 in sen2index_memo:
        tokenized_text_2 = sen2token_memo[sent2]
        indexed_tokens_2 = sen2index_memo[sent2]
    else:
        tokenized_text_2 = tokenizer.tokenize(' '+sent2+' ')
        indexed_tokens_2 = tokenizer.encode(' '+sent2+' ')[1:-1]
        sen2token_memo[sent2] = tokenized_text_2
        sen2index_memo[sent2] = indexed_tokens_2
    
    # print(tokenized_text, tokenized_text_1, tokenized_text_2)
    CLSID = 101
    SEPID = 102
    if 'deberta' in model_type:
        CLSID = 1
        SEPID = 2
    # try:
    tokenized_text = ['[CLS]'] + tokenized_text_1 + ['[SEP]'] + tokenized_text_2 + ['[SEP]']
    indexed_tokens = [CLSID] + indexed_tokens_1 + [SEPID] + indexed_tokens_2 + [SEPID]
    assert len(tokenized_text) == len(indexed_tokens)
    # except Exception as e:
    #     print(tokenized_text, tokenized_text_1, tokenized_text_2)
    #     print(indexed_tokens, indexed_tokens_1, indexed_tokens_2)
    #     raise e

    if len(tokenized_text) > 500:
        return None, None
    # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
    segments_ids = []
    sep_flag = False
    for i in range(len(indexed_tokens)):
        if 'roberta' in model_type and tokenized_text[i] == '</s>' and not sep_flag:
            segments_ids.append(0)
            sep_flag = True
        elif 'bert-' in model_type and tokenized_text[i] == '[SEP]' and not sep_flag:
            segments_ids.append(0)
            sep_flag = True
        elif sep_flag:
            segments_ids.append(1)
        else:
            segments_ids.append(0)
    return indexed_tokens, segments_ids

TOKENIZER = None
MODEL_TYPE = None
def job_build_batch(pair):
    sent1, sent2 = pair 
    ids, segs = get_pair_input(TOKENIZER,sent1,sent2,MODEL_TYPE)
    # if ids is None or segs is None: continue
    # token_id_list.append(ids)
    # segment_list.append(segs)
    # attention_masks.append([1]*len(ids))
    return ids, segs, [1]*len(ids), len(ids)

def build_batch(tokenizer, text_list, model_type):
    token_id_list = []
    segment_list = []
    attention_masks = []
    longest = -1

    if n_pool > 0:
        global TOKENIZER
        TOKENIZER = tokenizer
        global MODEL_TYPE
        MODEL_TYPE = model_type
        pool = Pool(n_pool) 
        results = pool.map(job_build_batch, text_list)
        pool.close() 
        pool.join()
        token_id_list, segment_list, attention_masks, attention_masks_lens = zip(*results)
        token_id_list = list(token_id_list)
        segment_list = list(segment_list)
        attention_masks = list(attention_masks)
        longest = max(attention_masks_lens)
    else:
        for pair in text_list:
            sent1, sent2 = pair 
            ids, segs = get_pair_input(tokenizer,sent1,sent2,model_type)
            if ids is None or segs is None: continue
            token_id_list.append(ids)
            segment_list.append(segs)
            attention_masks.append([1]*len(ids))
            if len(ids) > longest: longest = len(ids)

    if len(token_id_list) == 0: return None, None, None

    # padding
    assert(len(token_id_list) == len(segment_list))
    for ii in range(len(token_id_list)):
        token_id_list[ii] += [0]*(longest-len(token_id_list[ii]))
        attention_masks[ii] += [0]*(longest-len(attention_masks[ii]))
        segment_list[ii] += [1]*(longest-len(segment_list[ii]))

    return token_id_list, segment_list, attention_masks

def tokenlist_pairs_padding(tokenlist_pairs):
    # tokenlist_pairs: [([ids of s1], [ids of s2]), ([ids of s3],[ids of s4])...]
    attention_masks = [[1]*(len(pair[0])+len(pair[1])+3) for pair in tokenlist_pairs]
    segment_list = [[0]*(len(pair[0])+2)+[1]*(len(pair[1])+1) for pair in tokenlist_pairs]
    token_id_list = [[101]+pair[0]+[102]+pair[1]+[102] for pair in tokenlist_pairs]
    longest = max(map(len, segment_list))

    assert(len(token_id_list) == len(segment_list))
    for ii in range(len(token_id_list)):
        token_id_list[ii] += [0]*(longest-len(token_id_list[ii]))
        attention_masks[ii] += [0]*(longest-len(attention_masks[ii]))
        segment_list[ii] += [1]*(longest-len(segment_list[ii]))

    return token_id_list, attention_masks, segment_list