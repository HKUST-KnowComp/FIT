# Rethinking Complex Queries on Knowledge Graphs with Neural Link Predictors

This repository is for implementation for the paper "Rethinking Complex Queries on Knowledge Graphs with Neural Link Predictors" of ICLR 2024. 
See the [openreview version](https://openreview.net/forum?id=1BmveEMNbG), or [arxiv version](https://arxiv.org/abs/2304.07063).


## 1 Preparation

### 1.1 Data Preparation
Please download the data from [here](https://drive.google.com/drive/folders/17bPr6_ESqh5D0LgWNgpE4mY8gpg2iC5o?usp=sharing), 
the data of three knowledge graphs can be downloaded separately and put it in the `data` folder.

A example data folder should look like this:
```
data/FB15k-237-EFO1
  - kgindex.json
  - train_kg.tsv
  - valid_kg.tsv
  - test_kg.tsv
  - train_qaa.json
  - valid_type0000_real_EFO1_qaa.json
  - valid_...
  - valid_type0000_real_EFO1_qaa.json
  - test_...
```
We note that there is only one file for training but multiple files for testing. 

Moreover, the four csv files containing the formulas used in training and testing should be directly put into the `data` folder:
```
data/FIT_finetune1p3in.csv
  - DNF_evaluate_EFO1.csv
  - FIT_quick_evaluate.csv
  - DNF_evaluate_c.csv
```

The data of formula type0000-type0013 are inherited from the original BetaE dataset and converted 
to the format that can be used in our experiment, type0014-type0023 are the ten new formulas sampled by ourself.

Moreover, one can also use our pipeline to sample query for their own after 
you have already downloaded the knowledge graph from above.

We give the example of sampling query for test from FB15k-237:

```angular2html
python sample_query.py --output_folder data/FB15k-237-EFO1 --data_folder data/FB15k-237-EFO1 --mode test
```

### 1.2 Finetuning the neurla link preictor

To train the model, first download checkpoint of pretrained neural link predictor 'pretrain_cqd.zip' from [here](https://drive.google.com/drive/folders/17bPr6_ESqh5D0LgWNgpE4mY8gpg2iC5o?usp=sharing).

Unzip the file, the folder `pretrain/cqd` should look like this:
```
pretrain/cqd/FB15k-237.pt
  - FB15k.pt
  - NELL.pt
```

Then you can finetune these models by running the following code, taking FB15k-237 as an example:

```
python QG_FIT.py --config config/real_EFO1/FIT_FB15k-237_EFO1.yaml
```



## 2. Reproduce the result of the paper.

### 2.1 Evaluation

When comes to the evaluation, you need to use the fine-tuned neural link predictor get in the first step, 
or you can download the fine-tuned model 'pretrain_FIT.zip' from [here](https://drive.google.com/drive/folders/17bPr6_ESqh5D0LgWNgpE4mY8gpg2iC5o?usp=sharing).
After downloading or finetuning, move the checkpoint into `pretrain/FIT` folder, the folder should look like this:
```
pretrain/FIT/FB15k-237/5000.ckpt
  - /FB15k/...
  - /NELL/...
```
If you fineutune the neural link predictor yourself, you should also put the checkpoint in the corresponding folder.
(Manually assign the path of the checkpoint in the config file for testing is also OK, in 'load'/'checkpoint_path' and 'load'/'step')

Then you can run the following code, taking FB15k-237 as an example:
```
python QG_FIT.py --config config/test_EFO1/FIT_FB15k-237_EFO1.yaml
```


We note that the matrix can be directly created by the finetuned neural link predictor, but it will take a long time 
when it is created at the first time, then it will be saved in the `sparse` folder.

An example of the `sparse` sub folder should look like this:
```
sparse/FB15k-237
  - finetune_1p3in.ckpt
```



### 2.2 Ablation Study
If you want to reproduce the ablation study of the influence of hyperparameter, you can make some adjustment as the following.

For different max enumeration:
```
python QG_FIT.py --config config/test_EFO1/FIT_FB15k-237_m5.yaml
python QG_FIT.py --config config/test_EFO1/FIT_FB15k-237_m15.yaml
```

For different epsilon, delta:
```
python QG_FIT.py --config config/test_EFO1/FIT_FB15k-237_EFO1_eps01.yaml
python QG_FIT.py --config config/test_EFO1/FIT_FB15k-237_EFO1_thres01.yaml
python QG_FIT.py --config config/test_EFO1/FIT_FB15k-237_EFO1_thres02.yaml
```


For different c_norm, it is a bit complicated as new neural link predictor should be finetuned first:
```
python QG_FIT.py --config config/real_EFO1/FIT_FB15k-237_Godel.yaml
```
After assigning the path of the finetuned model in the config file,
(you can also download the finetune neural link predictor [here](https://drive.google.com/drive/folders/17bPr6_ESqh5D0LgWNgpE4mY8gpg2iC5o?usp=sharing)) you can run the following code:

```
python QG_FIT.py --config config/test_EFO1/FIT_FB15k-237_Godel.yaml
```


## 3. Citing the paper

Please cite the paper if you found the resources in this repository useful.
```
@inproceedings{yin2023rethinking,
  title={Rethinking Complex Queries on Knowledge Graphs with Neural Link Predictors},
  author={Yin, Hang and Wang, Zihao and Song, Yangqiu},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2023}
}
```