# Global Explainability of GNNs via Logic Combination of Learned Concepts

This repo is the official implementation of [GLGExplainer](https://arxiv.org/abs/2210.07147) presented at ICLR2023.


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


### Running GLGExplainer