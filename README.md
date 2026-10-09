# Micrograd and makemore learning notebooks

> **Status:** Backprop and bigram are implemented, with known bugs still to fix. The MLP notebook is in progress. See the limitations below.

Python notebooks exploring scalar automatic differentiation, small neural networks, and character-level name models. These are tutorial learning exercises, not an original machine-learning framework or production-ready package.

## Files

- **backprop.ipynb**: scalar Value class, computation graphs, reverse-mode differentiation, Neuron/Layer/MLP classes, Graphviz visualization, and a small gradient-descent example.
- **makemore_bigram.ipynb**: character counts, a smoothed bigram probability table, name sampling, negative log likelihood, and a PyTorch neural bigram experiment.
- **makemore_mlp.ipynb**: work in progress; context-window dataset construction and an embedding lookup. A complete MLP and training loop are not implemented yet.
- **names.txt**: the 32,033-line names dataset used by the makemore notebooks. Its exact source and redistribution terms still need verification before this repo is published.

## Run locally

Install Python and the dependencies, then open Jupyter from this folder so relative dataset paths resolve:

```sh
python -m pip install -r requirements.txt
python -m notebook
```

Graph rendering additionally needs the Graphviz system application (`dot`) installed and available on PATH. PyTorch installation requirements depend on your platform.

## Current limitations

The original files and saved outputs were uploaded unchanged; the notebooks have not been verified with a clean end-to-end run.

- Backprop: division and reverse division do not preserve a valid gradient graph; reverse subtraction reverses the wrong operands. The tanh implementation can overflow. Training has no maximum iteration bound and logs every iteration; its saved output includes an interrupted run. Reset gradients appropriately before repeated backward passes.
- Bigram: the first cell uses `words` before it is loaded, so a fresh top-to-bottom run raises NameError. Load the dataset before deriving `chars`. Some examples assume a fixed alphabet size of 27 and a first word containing five character transitions.
- MLP: unfinished exploratory notebook with saved error output.

## Attribution

The micrograd implementation follows Andrej Karpathy's tutorial; the Graphviz helper is marked as copied in the notebook. The makemore notebooks follow the tutorial approach and are presented as learning exercises.

- [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd)
- [Andrej Karpathy's makemore](https://github.com/karpathy/makemore)
- [Neural Networks: Zero to Hero](https://github.com/karpathy/nn-zero-to-hero)

No claim of original authorship is made for tutorial-derived algorithms or copied helpers. Before making this repo public, confirm source attribution and redistribution permissions, fix the run-order/arithmetic issues, and add explanations and results you can defend.
