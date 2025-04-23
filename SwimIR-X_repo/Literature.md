### table of content

- [interview list](#interview-list)
- [SWIMIR](#swimir)
- [Gecko](#gecko)
- [E5](#e5)
- [Gemini](#gemini)
- [Beyond model and dataset](#beyond-model-and-dataset)

## interview list

Similar methods for multilingual dataset curation/models:

1. SWIM-IR (https://arxiv.org/abs/2311.05800)
2. Gecko (https://arxiv.org/abs/2403.20327)
3. E5 Multilingual (https://arxiv.org/abs/2402.05672)
4. Gemini Embeddings (https://arxiv.org/abs/2503.07891)
5. KaLM (https://arxiv.org/abs/2501.01028)
6. BGE-M3 (https://arxiv.org/abs/2402.03216)
7. mGTE (https://arxiv.org/abs/2407.19669)
8. Promptagator (https://arxiv.org/abs/2209.11755)
9. Best practices (https://arxiv.org/abs/2204.02363)
10. ByT5(https://arxiv.org/abs/2105.13626)
11. XLM-R(https://arxiv.org/abs/1911.02116)

Curated Datasets:

1. MIRACL (https://aclanthology.org/2023.tacl-1.63/)
2. NeuCLIR(https://arxiv.org/abs/2304.12367)
3. MKQA(https://aclanthology.org/2021.tacl-1.82/)
4. mMARCO(http://arxiv.org/abs/2108.13897)
5. JH-POLO(https://doi.org/10.48550/ARXIV.2305.00331)
6. TriviaQA(English only)(https://aclanthology.org/P17-1147/)
7. SQuAD(English)(https://rajpurkar.github.io/SQuAD-explorer/)
8. XQuAD(multi-lingual translated from SQuAD)(https://arxiv.org/abs/1910.11856)
9. MLQA(https://aclanthology.org/2020.acl-main.653/)
10. XNLI(https://arxiv.org/abs/1809.05053)
11. NLLB(https://arxiv.org/abs/2207.04672)
12. Mr. TyDi()
13. MS MARCO(English only)(https://arxiv.org/abs/1611.09268)
14. NQ(English only)(https://paperswithcode.com/dataset/natural-questions)
15. ELI5(English only)(https://aclanthology.org/P19-1346/)
16. Dureader(Chinese only)(https://aclanthology.org/W18-2605/)
17. FEVER(English only)
18. HotpotQA(English only)
19. Quora Duplicated questions(English only)
20. Mr. TyDi(https://aclanthology.org/2021.mrl-1.12/)

InPars

## SWIMIR

Curation dataset:

SwimIR(https://arxiv.org/abs/2311.05800): [XOR-Retrieve](https://aclanthology.org/2021.naacl-main.46/) (cross-lingual), MIRACL (monolingual) and XTREME-UP (cross-lingual)

Model:

SwimIR:

mContriever-X(backbone mT5+MS MARCO ), mDPR-X, SIWM-IR backbone(mT5)-> substitute backbone: ByT5-Base

base line: XOR-Retrieve: Dr.DECR, ColBERT, XQG, two-stage translation baseline: Google Translate and Opus-MT// MIRACL: BM25, mDPR and Hybrid(BM25 + mDPR, mColeBERT)

PLM: **XLM-R, mBERT, BERT, mT5** pretrain dataset **mC4**, finetune **MS MARCRO + XOR , SWIMIR**

Benchmark(**SwimIR XTREME-UP **randomly select -> finetune?)

## Gecko

A standout feature of Gecko is its compactness- despite using 256 embedding dimensions, it outerperforms all existing entries with 768 embeddings. Another key strength of Gecko is its versatility; it is designed to leverage LLMs to develop a general-purpose text embedding model.

Gecko is based on a 1.2B parameter pre-trained transformer language model that undergoes two additional training stages: pre-finetuning and fine-tuning

Pre-finetuning stage: 2 datasets(QA + text crawled online)

FRet: Two-step LLM Distillation: We begin by randomly selecting a topic from web, and then prompt a LLM to generate a specific task along with a qeury. The two-step process introduce diversity across different both topic and tasks, making FRet a broadly applicable dataset suitable for training general-purpose embedding models.

HN mining: query likelihood and relevance classification, RRF.

Model: Gecko(mono-lingual, but domonstrates strong transferabilit to multilingual tasks). We extend its capabilities by incorporating MIRACL training dataset into the training mixture.

Dataset: MIRACL, MS-MARCO

We can also use the LLM as a labeler to train our HN.

## E5

Three embedding models of different sizes(small/base/large) are provided. The training procedure involves pretraining on 1 billion multilingual text pairs, followed by fine-tuning on a combination of labeled datasets.

models:
E5 family

Curated dataset:
MS-MARCO Passage, MS-MARCO Document, NQ, TriviaQA, SQuAD, NLI, ELI5, NLLB, DuReader Retrieval, Fever, HotpotQA, Quora Duplicate Questions, Mr. TyDi, MIRACL

## Gemini

## Beyond model and dataset

[APPROXIMATE NEAREST NEIGHBOR NEGATIVE CONTRASTIVE LEARNING FOR DENSE TEXT RETRIEVAL](https://openreview.net/forum?id=zeFrfgyZln)

[asynchronously-updated approximate nearest neightbor](https://arxiv.org/abs/2007.00808)- A datastructure to select HN(efficient, suitable for a large corpus)
