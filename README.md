# Global Explainability of GNNs via Logic Combination of Learned Concepts

This repo is the official implementation of [GLGExplainer](https://arxiv.org/abs/2210.07147) accepted at ICLR2023.

To cite our work:


```
@article{azzolin2022global,
  title={Global explainability of gnns via logic combination of learned concepts},
  author={Azzolin, Steve and Longa, Antonio and Barbiero, Pietro and Liò, Pietro and Passerini, Andrea},
  journal={arXiv preprint arXiv:2210.07147},
  year={2022}
}
```


## Using GLGExplainer

### Requirements

To install the required packages, run:

```
conda env create -f env.yml
```

The [torch-explain](https://pypi.org/project/torch-explain/) package is also needed.

:warning: The [PR](https://github.com/pietrobarbiero/pytorch_explain/pull/4) has been merged. Make sure to have the latest version of `torch-explain`

```
pip install torch-explain
```

### Extracting local explanations

To extract local explanations you can use any available Local Explainer which meets the requirements described in our paper. For a list of most commonly used Local Explainers for GNNs you can check out [this](https://arxiv.org/abs/2210.15304) survey.

Local explanations can be saved in a sub-folder similarly as done in `local_explanations\PGExplainer\` and can be read by adding a custom function in `code\local_explanations.py` to properly read the explanations.

### Running GLGExplainer

You can either train a new instance of GLGExplainer, or run the pre-trained model, via the notebooks in the `code\` folder. 

A compact summary of *how to train* GLGExplainer is the following:

```python
import utils
import models
from local_explanations import *

# read local explanations
train_data = read_bamultishapes(split="TRAIN")
val_data   = read_bamultishapes(split="VAL")
test_data  = read_bamultishapes(split="TEST")

# group local explanations into a Dataset
dataset_train = utils.LocalExplanationsDataset(train_data, ...)
dataset_val   = utils.LocalExplanationsDataset(val_data, ...)
dataset_test  = utils.LocalExplanationsDataset(test_data, ...)

# build a PyG DataLoader
train_group_loader = utils.build_dataloader(dataset_train, ...)
val_group_loader   = utils.build_dataloader(dataset_val, ...)
test_group_loader  = utils.build_dataloader(dataset_test, ...)

# init the LEN
len_model    = models.LEN(hyper_params["num_prototypes"], 
                          hyper_params["LEN_temperature"], 
                          remove_attention=hyper_params["remove_attention"])

# init the Local Explanations Embedder
le_model     = models.LEEmbedder(num_features=hyper_params ["num_le_features"], 
                                 activation=hyper_params["activation"], 
                                 num_hidden=hyper_params["dim_prototypes"])

# init GLGExplainer
expl         = models.GLGExplainer(len_model, 
                                   le_model, 
                                   hyper_params=hyper_params,
                                   ...)

# train GLGExplainer
expl.iterate(train_group_loader, val_group_loader, plot=True)

# inspect embedding, prototypes, and logic formulas
expl.inspect(test_group_loader)
```
