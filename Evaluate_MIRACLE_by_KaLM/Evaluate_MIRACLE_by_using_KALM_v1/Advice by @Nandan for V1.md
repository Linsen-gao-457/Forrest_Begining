## Advice by @Nandan for ==V1==

Current issues: take the positive passages and embed them and retrieve the passage from the index.

### How to correct

In a typical IR setting, you need to embed the passages in a corpus (dataset which contains all information) - which you will embed all passages in your index, for example for yoruba your index should contain 49,043K passages and english should contain 32M (million) passages.

reproduce: use [pytrec_eval](https://github.com/beir-cellar/beir/blob/main/beir/retrieval/evaluation.py) to ensures standardization to rewrite the code to evaluate [mcontriver model](https://huggingface.co/datasets/miracl/miracl-corpus) on Yoruba in [MIRACL](https://huggingface.co/datasets/miracl/miracl-corpus) to valid my evaluation method( 0.415 nDCG@10 and 0.770 Recall@100) 

Evalue again

