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

To install the required packages, run

```
conda env create -f env.yml
```

The following package is also needed

:warning: The updated version of torch-explain with the code to support GLGExplainer is not yet available. In case you are interested to use the code ASAP, write me an email :) 

```
pip install torch-explain
```

### Extracting local explanations

To extract local explanations you can use any available Local Explainer which meets the requirements described in our paper. For a list of most commonly used Local Explainers for GNNs you can check out [this](https://arxiv.org/abs/2210.15304) survey.

Local explanations can be saved in a sub-folder similarly as done in `local_explanations\PGExplainer\` and can be read by adding a custom function in `code\local_explanations.py` to properly read the explanations.

### Running GLGExplainer

You can either train a new instance of GLGExplainer, or run the pre-trained model, via the notebooks in the `code\` folder. 

A compact summary of how to run GLGExplainer is the following:

```python
import utils
import models
from local_explanations import *

train_data = read_bamultishapes(split="TRAIN")
val_data = read_bamultishapes(split="VAL")
test_data = read_bamultishapes(split="TEST")

dataset_train = utils.LocalExplanationsDataset(train_data, ...)
dataset_val = utils.LocalExplanationsDataset(val_data, ...)
dataset_test = utils.LocalExplanationsDataset(test_data, ...)

train_group_loader = utils.build_dataloader(dataset_train, ...)
val_group_loader   = utils.build_dataloader(dataset_val, ...)
test_group_loader  = utils.build_dataloader(dataset_test, ...)

len_model    = models.LEN(hyper_params["num_prototypes"], 
                          hyper_params["LEN_temperature"], 
                          remove_attention=hyper_params["remove_attention"])
le_model     = models.LEEmbedder(num_features=hyper_params ["num_le_features"], 
                                 activation=hyper_params["activation"], 
                                 num_hidden=hyper_params["dim_prototypes"])

expl         = models.GLGExplainer(len_model, 
                                   le_model, 
                                   hyper_params=hyper_params,
                                   ...
                                  )
expl.iterate(train_group_loader, val_group_loader, plot=True)
expl.inspect(test_group_loader)
```
