# Introduction
This repository is used in our paper:  
  
<!-- [**Jointly Learning Aspect-Focused and Inter-Aspect Relations with Graph Convolutional Networks for Aspect Sentiment Analysis**](https://www.aclweb.org/anthology/2020.coling-main.13/) -->
**JointCL: A Joint Contrastive Learning Framework for Zero-Shot Stance Detection**
<br>
Bin Liang, Qinlin Zhu, Xiang Li, Min Yang, Lin Gui, Yulan He, Ruifeng Xu<sup>\*</sup>. *Proceedings of ACL 2022*

Please cite our paper and kindly give a star for this repository if you use this code.

## Requirements

* Python 3.6
* PyTorch 1.6.0
* faiss-gpu 1.7.1
* transformers 2.5.1


## Usage

* Install [faiss](https://github.com/facebookresearch/faiss) package.

## Training
* Train with command, optional arguments could be found in [run_semeval.py](/run_semeval.py) \& [run_vast.py](/run_vast.py) \& [run_wtwt.py](/run_wtwt.py)


* Run Semeval dataset: ```python ./run_semeval.py```

* Run VAST dataset: ```python ./run_vast.py```

* Run WTWT dataset: ```python ./run_wtwt.py```

## Reproduce

* Run python files in [run_checkpoints](/run_checkpoints), you can use the trained model for prediction, and the model can be downloaded from [Google drives](https://drive.google.com/drive/folders/1W-UIVfHVgsLycTZdEIb4gNhGKCBW2wKo?usp=sharing).

<!-- ## Citation

The BibTex of the citation is as follow:

```bibtex
@inproceedings{liang-etal-2020-jointly,
    title = "Jointly Learning Aspect-Focused and Inter-Aspect Relations with Graph Convolutional Networks for Aspect Sentiment Analysis",
    author = "Liang, Bin  and
      Yin, Rongdi  and
      Gui, Lin  and
      Du, Jiachen  and
      Xu, Ruifeng",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "International Committee on Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.coling-main.13",
    pages = "150--161",
    abstract = "In this paper, we explore a novel solution of constructing a heterogeneous graph for each instance by leveraging aspect-focused and inter-aspect contextual dependencies for the specific aspect and propose an Interactive Graph Convolutional Networks (InterGCN) model for aspect sentiment analysis. Specifically, an ordinary dependency graph is first constructed for each sentence over the dependency tree. Then we refine the graph by considering the syntactical dependencies between contextual words and aspect-specific words to derive the aspect-focused graph. Subsequently, the aspect-focused graph and the corresponding embedding matrix are fed into the aspect-focused GCN to capture the key aspect and contextual words. Besides, to interactively extract the inter-aspect relations for the specific aspect, an inter-aspect GCN is adopted to model the representations learned by aspect-focused GCN based on the inter-aspect graph which is constructed by the relative dependencies between the aspect words and other aspects. Hence, the model can be aware of the significant contextual and aspect words when interactively learning the sentiment features for a specific aspect. Experimental results on four benchmark datasets illustrate that our proposed model outperforms state-of-the-art methods and substantially boosts the performance in comparison with BERT.",
}
```
 -->

## Credits

* The code of this repository partly relies on [ASGCN](https://github.com/GeneZC/ASGCN) \& [ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch) \& [PCL](https://github.com/salesforce/PCL). 
* Here, I would like to express my gratitude to the authors of the [ASGCN](https://github.com/GeneZC/ASGCN) \& [ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch) \& [PCL](https://github.com/salesforce/PCL) repositories.

