# Text2Path: model for mapping  unrestricted text to knowledge graph entities

## Table of contents

1. [Replicate Results](#replicated-results)
2. [Usage](#usage)
3. [Citing](#citing)
4. [Licence](#licence)
5. [Contact info](#contact-info)

## Replicate Results

We provide pretrained models for text2edges(*)[last row of Table 2]. To replicate the results:

### Step 1:

```
$ cd ./Text2ath/Scripts/Model/Trained_Models/
```

Unzip the pretrained models in the directory.

### Step 2:

```
$ cd ./Text2Path/Scripts/Model/
$ sh get_reported_results.sh
```    

Code for the MS-LSTM model can be found [here](https://bitbucket.org/dimkart/ms-lstm/src/master/)

## Usage
### Step 1: Graph Preprocessing
Algorithm assumes that a graph is a [rooted tree](http://mathworld.wolfram.com/RootedTree.html) and it is represented as an edge list:

```
node1 node5
node2 node4
...
``` 
Furthermore each node in the graph must have a textual definition:

```
node1 text_def_1
node2 text_def_2
...
```

To preprocess a graph for a text2nodes model:

```
$ cd ./Text2Path/Scripts/Preprocessing/
$ sh make_path_node_representation_dataset.sh
```
To preprocess a graph for a text2edges model:

```
$ cd ./Text2Path/Scripts/Preprocessing/
$ sh make_artificial_vocab_representation_dataset.sh
```

One can get the pretrained word embeddings used in the experiments [here](https://github.com/commonsense/conceptnet-numberbatch)

### Step2 Train a Model:

To train a new model:

```
$ cd ./Text2Path/Scripts/Model/
$ python text_to_path_model.py --train_data <path_to_train_data> --augment_data <augment_data_file> --test_data <test_data_file> --checkpoint <save_model_file> --graph <graph_file> --is_train 1
```



## Citing

If you find this material useful in your research, please cite:

```
@InProceedings{prokhorov_etal:NAACL2019,
  author={Victor Prokhorov and Mohammad T. Pilehvar and Nigel Collier},
  title={Generating Knowledge Graph Paths from Textual Definitions using Sequence-to-Sequence Models},
  booktitle={Proceedings of the 2019 Annual Conference of the North American Chapter of the Association for Computational Linguistics},
  year={2019},
  month={June},
  address={Minneapolis, USA},
  publisher={Association for {C}omputational {L}inguistics}
}  
```

## Licence

The code in this repository is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License version 3 as published by the Free Software Foundation. The code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the [GNU General Public License](https://www.gnu.org/licenses/gpl-3.0.en.html) for more details.


## Contact info

For questions or more information please use the following:
* **Email:** vp361@cam.ac.uk 
