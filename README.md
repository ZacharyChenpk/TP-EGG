# TP-EGG
The repository contains the source code of the ACL 2023 long paper "From the One, Judge of the Whole: Typed Entailment Graph Construction with Predicate Generation". [Arxiv](https://arxiv.org/abs/2306.04170)

## How to run
1. Download the evaluation [repository](https://github.com/mjhosseini/entgraph_eval.git) and the dataset for later usage:

```
git clone https://github.com/mjhosseini/entgraph_eval.git
cd TP-EGG/
wget https://dl.dropboxusercontent.com/s/j7sgqhp8a27qgcf/gfiles.zip
unzip gfiles.zip
rm gfiles.zip
mv gfiles/ .. 
ls ../gfiles/ent/
ls ../entgraph_eval/
```

2. Run the training script to get the predicate generator $G$ based on T5-large, or directly download the [checkpoint](https://drive.google.com/file/d/1eMN2Bl0JnCd2Zs6CNE-l7AkKt6Ro6w8T/view?usp=sharing): 

```
python t5_train.py --t5_size large --lr 1e-3
stat t5_tuned_large_0.001_reannofix.pth
```

3. Run the predicate generator to generate novel predicate sets:

```
python t5_predgen.py --beam 50 --keep 5000 --filter_level 0 --t5_size large --model_path t5_tuned_large_0.001_reannofix.pth
```

4. Run the training script to get the edge selictor $M$ based on BERT, or directly download the [checkpoint](https://drive.google.com/file/d/1RPD_vN9Rl3_J4teta2kij-obcy19vBHE/view?usp=drive_link):

```
mkdir sent_matchers
python -u sent_matcher_ball.py --lr 1e-5 --dnet_lr 5e-4 --n_epoch 300 --pos_repeat 5 --alltest 1 --dist_last exp --d_middim 4 --embmod_dim 16
stat sent_matchers/ball2_alltest_bertbase_0.9_1e-05_0.0005_5_exp_4_em16_best.pth.tar
```

5. Download the edge weight calculator $W$ [checkpoint](https://drive.google.com/file/d/1gTajRfdz83oCv0Q5zz5wSMoX_Y6S5AA5/view?usp=drive_link) re-implemented based on [EGT2](https://github.com/ZacharyChenpk/EGT2/tree/main):

```
mkdir deberta_tars
mv deberta0.8_12_1e-05_1_reannofix_best.pth.tar deberta_tars/
stat deberta_tars/deberta0.8_12_1e-05_1_reannofix_best.pth.tar
```

6. Run the local graph generation script to get the output EG, and you can change the parameters of Line 32-43 in `sent_matcher_modifier_ball_for_t5predgen.py`:

```
python sent_matcher_modifier_ball_for_t5predgen.py
ls ../gfiles/typedEntGrDir_sent_matcher_t5gen_b50k5000fil02nei_large_tunedlarge0.001reannofix_ball2m_1e-5_5e-4_5_exp_4_em16_f2e7_mNone_ty2/
```

And we upload the final [graphs](https://huggingface.co/zacharyc/TPEGG_LH/tree/main) for future research. After downloading corresponding files, get the graphs by:

```
cat TPEGG_LH_split.z* > graph.zip
unzip graph.zip
```

7. Copy the relaxed evaluation script to the evaluation repo, and get the evaluation results:

```
cp eval_with_sent.py ../entgraph_eval/evaluation/
cd ../entgraph_eval/evaluation/
python eval.py --gpath typedEntGrDir_sent_matcher_t5gen_b50k5000fil02nei_large_tunedlarge0.001reannofix_ball2m_1e-5_5e-4_5_exp_4_em16_f2e7_mNone_ty2 --test_reanno --sim_suffix _sim.txt --method TPEGG_LH_reannofix --CCG 1 --typed 1 --supervised 0 --oneFeat 1 --useSims 0 --featIdx 0 --exactType --backupAvg --write
cd ../../TP-EGG
python eval_curvefill.py
```