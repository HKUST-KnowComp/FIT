# On Existential First Order Queries Inference on Knowledge Graphs

This repository is for implementation for the paper "On Existential First Order Queries Inference on Knowledge Graphs".

See the arXiv version [here](https://arxiv.org/abs/2304.07063).

## 1 Preparation
Firstly, make the directory of `data` and `sparse` in the root directory of this repository.
```angular2html
mkdir data
mkdir sparse
```
### 1.1 Data Preparation
Please download the data from [here](https://drive.google.com/drive/folders/17bPr6_ESqh5D0LgWNgpE4mY8gpg2iC5o?usp=sharing).




the data of three knowledge graphs can be downloaded separately and put it in the `data` folder and unzip it.

A example data folder should look like this:
```
data/FB15k-237-EFO1
  - kgindex.json
  - train_kg.tsv
  - valid_kg.tsv
  - test_kg.tsv
  - train-qaa.json
  - valid-qaa.json
  - test-qaa.json
  - test_real_EFO1_qaa.json
```

where only the `test_real_EFO1_qaa.json` is used for the real EFO1 experiment. Other data are inherited from the original BetaE dataset 
and converted to the format that can be used in our experiment.

### 1.2 Matrix Creation

The matrix that has been used in the paper can also be downloaded from [here](https://drive.google.com/drive/folders/17bPr6_ESqh5D0LgWNgpE4mY8gpg2iC5o?usp=sharing), 
where contains the matrix used for three knowledge graphs. We contain multiple checkpoint for each knowledge graph.

It should be put in the `sparse` folder and unzipped.

An example of the `sparse` sub folder should look like this:
```
sparse/237
  - torch_0.005_0.001.ckpt
```

## 2. Reproduce the result of the paper.

### 2.1 In real EFO1 dataset

For the reproduction of the experiment on FB15k-237 and FB15k in paper, run the following code:
```
## python solve_EFO1.py --ckpt 'sparse/237/torch_0.005_0.001.ckpt' --data_folder data/FB15k-237-EFO1
## python solve_EFO1.py --ckpt 'sparse/FB15k/torch_0.005_0.001.ckpt' --data_folder data/FB15k-EFO1
```

In case you have problem with your gpu memory, for example, for example, the experiments on the NELL dataset, you can run the following code:
```
## python solve_EFO1.v2.py --batch_size 1 --ckpt 'sparse/NELL/torch_0.001_0.001.ckpt' --data_folder data/NELL-EFO1
```

### 2.1 In BetaE dataset.

Simple changing the data_type to BetaE, you can reproduce the result in BetaE dataset, taking FB15k-237 as an example:
```
python solve_EFO1.py --ckpt 'sparse/237/torch_0.005_0.001.ckpt' --data_folder data/FB15k-237-EFO1 --data_type BetaE
```

### 2.2 Ablation Study
If you want to reproduce the ablation study of the influence of hyperparameter, you can make some adjustment as the following.
For different c_norm:
```
## python solve_EFO1.py --c_norm Godel
```
For different max enumeration:
```
## python solve_EFO1.py --max 5
## python solve_EFO1.py --max 20
```

For different epsilon, delta:
```
## python solve_EFO1.py --ckpt 'sparse/237/torch_0.01_0.001.ckpt'
## python solve_EFO1.py --ckpt 'sparse/237/torch_0.01_0.01.ckpt'
```

## 3. Citing the paper

Please cite the paper if you found the resources in this repository useful.

```
@article{yin2023existential,
  title={On Existential First Order Queries Inference on Knowledge Graphs},
  author={Yin, Hang and Wang, Zihao and Song, Yangqiu},
  journal={arXiv preprint arXiv:2304.07063},
  year={2023}
}