# Documentation

### Setup

This repository is tested on Python 3.8+. First, you should install a virtual environment:

```bash
python3 -m venv .venv/gdsr
source .venv/gdsr/bin/activate
```

Then, you can install all dependencies:

```bash
pip install -r requirements.txt
```

## Preliminaries

#### Domain-adaptative pretraining

Due to the specificity of the legal language the article encoder has to deal with, we continue pre-training a [CamemBERT](https://huggingface.co/camembert-base) checkpoint on [BSARD](https://huggingface.co/datasets/maastrichtlawtech/bsard) statutory articles to adapt it to the target legal domain. You can use the following command to perform the domain adaptation:

```bash
bash scripts/run_mlm_training.sh
```

#### Domain-specific augmentation

We propose to augment [BSARD](https://huggingface.co/datasets/maastrichtlawtech/bsard) with synthetic domain-targeted queries using a [mT5 model](https://huggingface.co/doc2query/msmarco-french-mt5-base-v1) fine-tuned on general domain data from the French [mMARCO](https://huggingface.co/datasets/unicamp-dl/mmarco) dataset. You can use the following command to perform the data augmentation:

```bash
bash scripts/augment_data.sh $method
```
where `$method` is either "back-translation" or "query-generation".

#### Hard negatives generation

In addition to in-batch negatives, we also use BM25 negatives during training as hard negatives, i.e., the top articles returned by BM25 that are not relevant to the question. To generate a set of five BM25 negatives for each query, you can run the following command:

```bash
bash scripts/generate_bm25_negatives.sh
```

#### Legislative graph creation

You can create the legislative graph of [BSARD](https://huggingface.co/datasets/maastrichtlawtech/bsard) by running the following command:

```bash
bash scripts/create_graph.sh
```

## Training

In order to train G-DSR, you first have to train the dense statute retriever (DSR), which learns high-quality low-dimensional embedding spaces for questions and articles so that relevant question-article pairs appear closer than irrelevant ones in those spaces. You can perform the training by running:

```bash
bash scripts/run_contrastive_training.sh biencoder
```

Once the dense retriever is trained, we use it to train the legislative graph encoder (LGE), which aims to enrich article representations given by the trained retriever's article encoder by fusing information from a legislative graph. You can perform the training by running:

```bash
bash scripts/run_contrastive_training.sh gnn
```

Additionnally, you can perform hyperparameter tuning with [Weights & Biases](https://wandb.ai/site) on both models by running the following commands:

```bash
wandb sweep src/config/<sweep_file>.yaml   #<sweep_file> in ['sweep_biencoder', 'sweep_gnn']
wandb agent <USERNAME/PROJECTNAME/SWEEPID>
```

## Evaluation

In order to evaluate DSR only, you can run the following command:

```bash
bash scripts/run_evaluation.sh biencoder
```

To evaluate our ultimate G-DSR model, run:

```bash
bash scripts/run_evaluation.sh gnn
```

#### Baselines

We compare our approach against three strong retrieval systems: BM25, [docT5query](https://huggingface.co/doc2query/msmarco-french-mt5-base-v1) and [DPR](https://huggingface.co/etalab-ia/dpr-ctx_encoder-fr_qa-camembert). You can evaluate those approaches by running the following commands, respectively:

```bash
bash scripts/run_bm25.sh
bash scripts/run_doc2query.sh
bash scripts/run_evaluation.sh #after changing BIENCODER_CKPT to the corresponding finetuned checkpoint
```

Note that you can also find the optimal BM25 hyperparameters by running:

```bash
bash scripts/tune_bm25.sh
```
