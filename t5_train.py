from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import os
from random import random, shuffle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--t5_size', type=str,
                    help="",
                    default='small')
parser.add_argument("--lr", type=float,
                    help="",
                    default=1e-3)
args = parser.parse_args()
print(args)

tokenizer = T5Tokenizer.from_pretrained("t5-"+args.t5_size)
model = T5ForConditionalGeneration.from_pretrained("t5-"+args.t5_size)
optimizer = torch.optim.AdamW(params=[p for p in model.parameters() if p.requires_grad], lr=args.lr)
EPOCHES = 500

input_sequences = []
output_sequences = []
test_input_sequences = []
test_output_sequences = []
max_source_length = 128
max_target_length = 32
conj_name = ''

with open('dev_pos_corpus_fort5_reanno.txt', 'r') as f:
# with open('sherliic_dev_pos_corpus_fort5.txt', 'r') as f:
    for l in f.readlines():
        inputtext, label = l.strip().split('\t')
        if random() < 0.8:
            input_sequences.append(inputtext)
            output_sequences.append(label)
        else:
            test_input_sequences.append(inputtext)
            test_output_sequences.append(label)

encoding = tokenizer(
    input_sequences,
    padding="longest",
    max_length=max_source_length,
    truncation=True,
    return_tensors="pt",
)
input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

target_encoding = tokenizer(
    output_sequences, padding="longest", max_length=max_target_length, truncation=True
)
labels = target_encoding.input_ids
labels = torch.tensor(labels)
labels[labels == tokenizer.pad_token_id] = -100

test_encoding = tokenizer(
    test_input_sequences,
    padding="longest",
    max_length=max_source_length,
    truncation=True,
    return_tensors="pt",
)
test_input_ids, test_attention_mask = test_encoding.input_ids, test_encoding.attention_mask

# encode the targets
test_target_encoding = tokenizer(
    test_output_sequences, padding="longest", max_length=max_target_length, truncation=True
)
test_labels = test_target_encoding.input_ids
test_labels = torch.tensor(test_labels)
test_labels[test_labels == tokenizer.pad_token_id] = -100

test_loss = model(input_ids=test_input_ids, attention_mask=test_attention_mask, labels=test_labels).loss.detach().item()
print('testloss', test_loss)

lowest = test_loss
tolerant = 10
no_inc = 0
best_param = model.state_dict()

nbatches = 2

for e in range(EPOCHES):
    optimizer.zero_grad()
    model.zero_grad()
    bsz = (len(input_ids) // nbatches)+1
    for i in range(nbatches):
        loss = model(input_ids=input_ids[bsz*i:bsz*(i+1)], attention_mask=attention_mask[bsz*i:bsz*(i+1)], labels=labels[bsz*i:bsz*(i+1)]).loss
        print(e, loss)
        loss.backward()
    optimizer.step()

    test_loss = model(input_ids=test_input_ids, attention_mask=test_attention_mask, labels=test_labels).loss.detach().item()
    print('testloss', test_loss)
    if test_loss < lowest:
        lowest = test_loss
        best_param = model.state_dict()
        no_inc = 0
    else:
        no_inc += 1
        if no_inc > tolerant:
            print('stop')
            break

torch.save(best_param, 't5_tuned_'+args.t5_size+'_'+str(args.lr)+'_reannofix.pth')
