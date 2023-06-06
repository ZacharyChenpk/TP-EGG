from scipy.sparse import data
import torch
import shutil
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from random import shuffle
import time
from tqdm import tqdm
import argparse
from transformers import BertModel, BertTokenizer
import os

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int,
                    help="GPU device number",
                    default=0)
parser.add_argument("--lr", type=float,
                    help="learning rate",
                    default=1e-5)
parser.add_argument("--dnet_lr", type=float,
                    help="d_net learning rate",
                    default=1e-4)
parser.add_argument("--n_epoch", type=int,
                    help="max number of epochs",
                    default=100)
parser.add_argument("--tolerant", type=int,
                    help="tolerant",
                    default=10)
parser.add_argument("--ratio", type=float,
                    help="train-test ratio",
                    default=0.9)
parser.add_argument("--pos_repeat", type=int,
                    help="repeat how much times for positive data",
                    default=1)
parser.add_argument("--alltest", type=int,
                    help="",
                    default=0)
parser.add_argument("--dist_last", type=str,
                    help="exp, sqr",
                    default='exp')
parser.add_argument("--d_middim", type=int,
                    help="",
                    default=16)
parser.add_argument("--embmod_dim", type=int,
                    help="",
                    default=None)

args = parser.parse_args()
print(args)

assert args.pos_repeat >= 1
torch.cuda.set_device(args.device)
ratio = args.ratio
lr = args.lr # 1e-5 originally

model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('/home/chenzhb/bert-base-uncased-vocab.txt')
d_net = torch.nn.Sequential(
    torch.nn.Linear(768, args.d_middim),
    torch.nn.ReLU(),
    torch.nn.Linear(args.d_middim, 1)
)
model.cuda()
d_net.cuda()

prefix = 'sent_matchers/ball2_'
if args.alltest > 0:
    prefix = 'sent_matchers/ball2_alltest_'
checkpoint_file = prefix + 'bertbase_'+str(ratio)+'_'+str(lr)+'_'+str(args.dnet_lr)+'_'+str(args.pos_repeat)+'_'+str(args.dist_last)+'_'+str(args.d_middim)+'_checkpoint.pth.tar'
best_file = prefix + 'bertbase_'+str(ratio)+'_'+str(lr)+'_'+str(args.dnet_lr)+'_'+str(args.pos_repeat)+'_'+str(args.dist_last)+'_'+str(args.d_middim)+'_best.pth.tar'
if args.embmod_dim is not None:
    checkpoint_file = prefix + 'bertbase_'+str(ratio)+'_'+str(lr)+'_'+str(args.dnet_lr)+'_'+str(args.pos_repeat)+'_'+str(args.dist_last)+'_'+str(args.d_middim)+'_em'+str(args.embmod_dim)+'_checkpoint.pth.tar'
    best_file = prefix + 'bertbase_'+str(ratio)+'_'+str(lr)+'_'+str(args.dnet_lr)+'_'+str(args.pos_repeat)+'_'+str(args.dist_last)+'_'+str(args.d_middim)+'_em'+str(args.embmod_dim)+'_best.pth.tar'
    embmod_net = torch.nn.Sequential(
        torch.nn.Linear(768, args.embmod_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(args.embmod_dim, args.embmod_dim)
    )
    embmod_net.cuda()
else:
    embmod_net = None

print(os.getpid(), checkpoint_file, best_file)

def checkpoint_save(epoch, mymodel, d_net, optimizer, best_pred, is_best, embmod_net=None):
    state = {'epoch': epoch + 1,
            'state_dict': mymodel.state_dict(),
            'd_net': d_net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_pred': best_pred}
    # torch.save(state, checkpoint_file)
    if embmod_net is not None:
        state['embmod_net'] = embmod_net.state_dict()
    if is_best:
        # shutil.copyfile(checkpoint_file, best_file)
        torch.save(state, best_file)

def checkpoint_load(model, d_net, optimizer, embmod_net=None):
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['state_dict'])
    d_net.load_state_dict(checkpoint['d_net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if embmod_net is not None:
        embmod_net.load_state_dict(checkpoint['embmod_net'])
    return checkpoint['epoch']

param_list = [
    {'params': [p for n,p in model.named_parameters() if p.requires_grad], 'lr': args.lr}, 
    {'params': [p for n,p in d_net.named_parameters() if p.requires_grad], 'lr': args.dnet_lr}
]
if args.embmod_dim is not None:
    param_list.append({'params': [p for n,p in embmod_net.named_parameters() if p.requires_grad], 'lr': args.dnet_lr})
optimizer = torch.optim.AdamW(params=param_list)
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)
for name, param in d_net.named_parameters():
    if 'weight' in name:
        torch.nn.init.xavier_uniform_(param)
    if param.requires_grad:
        print(name)

with open('dev_corpus_reanno.txt', 'r') as f_corpus:
# with open('sherliic_dev_corpus.txt', 'r') as f_corpus:
    data_list = list(f_corpus.readlines())
data_list = [a.strip().split('\t') for a in data_list]
shuffle(data_list)
train_num = int(ratio*len(data_list))
train_data = data_list[:train_num]
test_data = data_list[train_num:]
pos_data = [a for a in train_data if a[2]=='1']
if args.pos_repeat > 1:
    print('pos data len', len(pos_data))
    train_data = train_data + pos_data * (args.pos_repeat-1)
    shuffle(train_data)
train_X = [(a[0],a[1]) for a in train_data]
train_Y = torch.FloatTensor([int(a[2]) for a in train_data]).cuda()
test_X = [(a[0],a[1]) for a in test_data]
test_Y = torch.FloatTensor([int(a[2]) for a in test_data]).cuda()
bce_loss = nn.BCELoss()

# P = sigmoid((dq-d)/2dp)
def model_forward(batch, model, d_net, tokenizer, embmod_net=None):
    left_inputs = tokenizer([ll[0] for ll in batch], return_tensors="pt", padding=True)
    right_inputs = tokenizer([ll[1] for ll in batch], return_tensors="pt", padding=True)
    for key in left_inputs.keys():
        left_inputs[key] = left_inputs[key].cuda()
        right_inputs[key] = right_inputs[key].cuda()
    outputs = model(**left_inputs)
    left_pl = outputs.pooler_output
    outputs = model(**right_inputs)
    right_pl = outputs.pooler_output
    dp = d_net(left_pl).squeeze(1)
    dq = d_net(right_pl).squeeze(1)
    if args.embmod_dim is not None:
        left_pl = embmod_net(left_pl)
        right_pl = embmod_net(right_pl)
    if args.dist_last == 'exp':
        dp = dp.exp()
        dq = dq.exp()
    elif args.dist_last == 'sqr':
        dp = dp**2
        dq = dq**2
    else:
        raise Exception('Not Implemented')
    d = (left_pl-right_pl).norm(p=2,dim=1)
    # probs = (dq-d)/(2*dp)
    probs = 2*(dq-d)/dp
    probs = probs.sigmoid()
    return probs

def train_one_epoch(model, tokenizer, d_net, optimizer, train_X, train_Y, test_X, test_Y, embmod_net=None):
    model.train()
    stt = time.time()
    total_loss = 0
    # bs = 512
    bs = 256
    for batch_idx in tqdm(range(0,len(train_X),bs), disable=True,desc='Training'):
        model.zero_grad()
        optimizer.zero_grad()
        train_X_batch = train_X[batch_idx:batch_idx+bs]
        probs = model_forward(train_X_batch, model, d_net, tokenizer, embmod_net)
        loss = bce_loss(probs, train_Y[batch_idx:batch_idx+bs])
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print('total loss', total_loss, 'time', time.time()-stt)
    tp, tr, tf, ta = model_eval(model, d_net, tokenizer, train_X, train_Y, embmod_net)
    print('train P', tp, 'R', tr, 'F', tf, 'A', ta)

    return model_eval(model, d_net, tokenizer, test_X, test_Y, embmod_net)

def model_eval(model, d_net, tokenizer, test_X, test_Y, embmod_net=None):
    model.eval()
    with torch.no_grad():
        test_probs = None
        bbsz = (len(test_X) // args.pos_repeat)+1
        for b in range(args.pos_repeat):
            if test_probs is None:
                test_probs = model_forward(test_X[b*bbsz:(b+1)*bbsz], model, d_net, tokenizer, embmod_net)
            else:
                test_probs = torch.cat([test_probs, model_forward(test_X[b*bbsz:(b+1)*bbsz], model, d_net, tokenizer, embmod_net)], dim=0)
        # test_pred_Y = test_probs.argmax(dim=1)

        print(test_probs[:10], test_Y[:10])
        TP = ((test_probs > 0.5) & (test_Y == 1)).cpu().sum().item()
        TN = ((test_probs < 0.5) & (test_Y == 0)).cpu().sum().item()
        FN = ((test_probs <= 0.5) & (test_Y == 1)).cpu().sum().item()
        FP = ((test_probs >= 0.5) & (test_Y == 0)).cpu().sum().item()
        print(TP, TN, FN, FP)
        if TP + FP == 0:
            FP = 1
        p = TP / (TP + FP)
        r = TP / (TP + FN)
        if p + r == 0:
            F1 = 0
        else:
            F1 = 2 * r * p / (r + p)
        acc = (TP + TN) / (TP + TN + FP + FN)
        return p, r, F1, acc

n_epoch = args.n_epoch
p, r, F1, acc = model_eval(model, d_net, tokenizer, test_X, test_Y, embmod_net)
print('original','p:',p,'r:',r,'F1:',F1,'acc:',acc)

bestF1 = F1
best_param = model.state_dict()
tolerant = args.tolerant
no_inc = 0

for e in range(n_epoch):
    p, r, F1, acc = train_one_epoch(model, tokenizer, d_net,  optimizer, train_X, train_Y, test_X, test_Y, embmod_net)
    print('epoch:',e,'p:',p,'r:',r,'F1:',F1,'acc:',acc)
    checkpoint_save(e, model, d_net, optimizer, F1, (F1 > bestF1), embmod_net)
    if F1 <= bestF1:
        no_inc = no_inc + 1
        if no_inc == tolerant:
            print('Untolerable!')
            break
    else:
        bestF1 = F1
        no_inc = 0