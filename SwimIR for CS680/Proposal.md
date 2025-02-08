# reproduce the SWIM-IR model

Download the SWIM-IR dataset (monolingual for now) - mine hard negatives (not done till now) and improve the dataset. Next would be to fine-tune a suitable (multilingual) model using the SWIM-IR dataset you created.

Why we fine-tune a multilingual model using the improved SWIM-IR dataset?

1. For cross-Lingual retrieval needs: in future, this model can use French dataset to fine-tune, thus we can achieve two languages retrieval: use French query to find French document or English document?
2. Some other languages will learn the retrieval pattern in English; for this, even if fine-tuned only on English SWIM-IR dataset, the model can still learn retrieval pattern can be applied to other languages.

 

SWIM-IR dataset was constructed synthetically using PaLM (old google LLM) model. All queries there are synthetically generated. The advantage of cross-lingual data is alignment between languages, whereas monolingual data is alignment within a language. A realistic model needs to be great in both scenarios, i.e., if you check Cohere embed v3 which is multilingual (both monolingual and cross-lingual).The backstory is that the model was developed during my internship at Google, now due to company policies, I was unable to release the trained model. My ==motivation for you to work on this project== is to develop an open-source reproduction of the private model. You can see the scores which it achieves on datasets such as MIRACL, XOR-Retrieve in the paper: https://arxiv.org/pdf/2311.05800