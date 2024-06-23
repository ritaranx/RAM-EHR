# RAM-EHR

This is the code for our paper [RAM-EHR: Retrieval Augmentation Meets Clinical Predictions on Electronic Health Records
](https://arxiv.org/abs/2403.00815), published in ACL 2024.

## Data
**[IMPORTANT, PLEASE READ!]**
In order to facilitate the reproducibility, we provide a small synthetic subset in data for MIMIC-III dataset in `mimic` folder (`hyperedges-mimic-text-{train/valid/test}-example.jsonl`). We include them in this repo only to show the format of the two datasts we used. Thus, their experimental results should NOT reflect the performance we report in the paper.

## Knowledge Sources
We provide some of the publicly available sources in `mesh.txt` as well as the code for querying drugbank and pubmed database in `query_drugbank.py` and `query_pubmed.py`, respectively. The processed information for each medical code can be found in the `mimic` and `cradle` folder. The drugbank information can be accessed at [this link](https://go.drugbank.com/releases/latest). 

## Generating Knowledge Summaries using GPT-3.5
Please refer to `prompt_gpt_def.py` for details.

## Code
For graph neural network component, please follow our previous work on [hypergraph neural networks for EHR prediction](https://github.com/ritaranx/CACHE) for details. We have also provided the code of using language models for prediction in `model_src` folder.


## Citation
If you find this paper useful for your research, please cite the following in your publication. Thanks!


```
@inproceedings{xu2024ram,
  title={RAM-EHR: Retrieval Augmentation Meets Clinical Predictions on Electronic Health Records},
  author={Xu, Ran and Shi, Wenqi and Yu, Yue and Zhuang, Yuchen and Jin, Bowen and Wang, May D and Ho, Joyce C and Yang, Carl},
  booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics},
  year={2024}
}
```
