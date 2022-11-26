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

## Reproduce & Peformance

* Due to the small number of dataset samples (especially SEM16), the performance gap between different seeds will vary greatly, please tune the parameter of *--seed* for better performance.
* We have provided checkpoints that are superior or equal to the performance reported in the paper. 
* Please Run python files in [run_checkpoints](/run_checkpoints), you can use the trained model for prediction, and the model can be downloaded from [Google drives](https://drive.google.com/drive/folders/1W-UIVfHVgsLycTZdEIb4gNhGKCBW2wKo?usp=sharing).
* We also use 5 random seeds to run the code directly without any other tuning parameters. The performance is as follows:
<!--
    |Dataset | Task | Target | Reported | Checkpoint | seed1 | seed2 | seed3 | seed4 | seed5 | Mean | Max | Gap |
    | --------   | -----   |--------   | -----   |--------   | --------   | -----   |--------   | -----   |--------   |--------   | -----   |--------   |
    | Vast | Zero-shot | - | 72.3 |  72.4 | 70.6 | 71.3 | 72.4 | 72.0 | 71.3 | 71.5 | 72.4 | +0.1|
-->
    

<table border="1" width="500px" cellspacing="10">
	<tr>
		<th align="center">Task</th>
		<th align="center">Dataset</th>
		<th align="center">Target</th>
		<th align="center">Reported</th>
		<th align="center">Checkpoint</th>
		<th align="center">seed1</th>
		<th align="center">seed2</th>
		<th align="center">seed3</th>
		<th align="center">seed4</th>
		<th align="center">seed5</th>
		<th align="center">Mean</th>
		<th align="center">Max</th>
		<th align="center">Gap</th>
	</tr>
	<tr>
		<td rowspan="11" align="center">Zero-shot</td>
		<td>VAST</td><td> - </td><td>72.3</td><td>72.4</td><td>70.6</td><td>71.3</td><td>72.4</td><td>72.0</td><td>71.3</td><td>71.5</td><td>72.4</td><td>+0.1</td>
	</tr>
	<tr>
		<td rowspan="6" align="center">SEM16</td>
		<td> DT </td><td>50.5</td><td>50.9</td><td>46.0</td><td>40.6</td><td>45.6</td><td>48.4</td><td>50.2</td><td>46.2</td><td>50.2</td><td>-0.3</td>
	</tr>
	<tr>
		<td> HC </td><td>54.8</td><td>56.4</td>
		<td>50.7</td> <td>56.4</td> <td>45.7</td> <td>55.9</td> <td>51.3</td> 
		<td>52.0</td> <td>56.4</td> <td>+1.6</td>
	</tr>
	<tr>
		<td> FM </td> <td>53.8</td> <td>54.2</td>
		<td>50.9</td> <td>51.1</td> <td>49.1</td> <td>49.8</td> <td>49.4</td> 
		<td>50.1</td> <td>51.1</td> <td>-2.7</td>
	</tr>
	<tr>
		<td> LA </td> <td>49.5</td> <td>55.5</td>
		<td>54.8</td> <td>54.3</td> <td>55.5</td> <td>51.3</td> <td>47.1</td> 
		<td>52.6</td> <td>55.5</td> <td>+6</td>
	</tr>
	<tr>
		<td> A </td> <td>54.5</td> <td>54.6</td>
		<td>55.1</td> <td>48.0</td> <td>60.0</td> <td>55.4</td> <td>55.2</td> 
		<td>54.7</td> <td>60.0</td> <td>+5.5</td>
	</tr>
	<tr>
		<td> CC </td> <td>39.7</td> <td>40.7</td>
		<td>31.9</td> <td>36.9</td> <td>39.7</td> <td>40.2</td> <td>28.3</td> 
		<td>35.4</td> <td>40.2</td> <td>+0.5</td>
	</tr>
	<tr>
		<td rowspan="4" align="center">WTWT</td>
		<td> CA </td> <td>72.4</td> <td>73.6</td>
		<td>72.5</td> <td>71.4</td> <td>73.3</td> <td>73.4</td> <td>74.9</td> 
		<td>73.1</td> <td>74.9</td> <td>+2.5</td>
	</tr>
	<tr>
		<td> CE </td> <td>70.2</td> <td>70.9</td>
		<td>70.1</td> <td>71.4</td> <td>70.4</td> <td>70.3</td> <td>70.3</td> 
		<td>70.1</td> <td>71.4</td> <td>+1.2</td>
	</tr>
	<tr>
		<td> AC </td> <td>76.0</td> <td>76.5</td>
		<td>75.0</td> <td>74.3</td> <td>77.3</td> <td>73.3</td> <td>75.6</td> 
		<td>75.1</td> <td>77.3</td> <td>+1.3</td>
	</tr>
	<tr>
		<td> AH </td> <td>75.2</td> <td>76.5</td>
		<td>76.2</td> <td>76.1</td> <td>76.0</td> <td>77.9</td> <td>78.0</td> 
		<td>76.8</td> <td>78.0</td> <td>+2.8</td>
	</tr>
	<tr>
		<td> Few-shot </td> <td>VAST</td>
		<td> - </td> <td>71.5</td> <td>71.6</td>
		<td>71.6</td> <td>71.9</td> <td>68.4</td> <td>66.1</td> <td>69.5</td> 
		<td>69.5</td> <td>71.9</td> <td>+0.4</td>
	</tr>
	<tr>
		<td rowspan="4" align="center"> Cross-target </td> <td rowspan="4" align="center">SEM16</td>
		<td> HC->DT </td> <td>52.8</td> <td>54.6</td>
		<td>42.9</td> <td>46.9</td> <td>48.1</td> <td>53.7</td> <td>54.2</td> 
		<td>49.2</td> <td>54.2</td> <td>+1.4</td>
	</tr>
	<tr>
		<td> DT->HC </td> <td>54.3</td> <td>55.4</td>
		<td>52.1</td> <td>55.8</td> <td>54.6</td> <td>47.8</td> <td>38.6</td> 
		<td>49.8</td> <td>55.8</td> <td>+1.5</td>
	</tr>
	<tr>
		<td> FM->LA </td> <td>58.8</td> <td>60.0</td>
		<td>49.8</td> <td>58.0</td> <td>58.3</td> <td>46.7</td> <td>45.7</td> 
		<td>51.7</td> <td>60.0</td> <td>-0.5</td>
	</tr>
	<tr>
		<td> LA->FM </td> <td>54.5</td> <td>54.8</td>
		<td>45.8</td> <td>41.8</td> <td>54.1</td> <td>36.2</td> <td>47.9</td> 
		<td>45.2</td> <td>54.1</td> <td>-0.4</td>
	</tr>
</table>


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

