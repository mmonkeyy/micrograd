# Micrograd and makemore notebooks

Learning exercises following Andrej Karpathy's micrograd and makemore tutorials. These notebooks cover scalar automatic differentiation and character-level name generation.

**Status:** Backprop and bigram code cells pass sequential-run verification. The MLP notebook is a partial exercise, not a finished model.

## Notebooks

| File | Contents |
| --- | --- |
| backprop.ipynb | Value graph, reverse-mode gradients, neuron/layer classes, and a small training example |
| makemore_bigram.ipynb | Count-based probabilities, name sampling, negative log likelihood, and neural bigram training |
| makemore_mlp.ipynb | Context-window dataset and embedding lookup; forward pass and training remain to be written |

## Run

Use Python 3.10 or newer. From this folder:

```sh
python -m pip install -r requirements.txt
python -m notebook
```

Run notebook cells in order. Graph rendering additionally needs the Graphviz system application (`dot`). The bigram examples assume lowercase English names and use `.` as a start/end token.

## Checks

```sh
python test_gradients.py
python verify_notebooks.py
```

The gradient checks compare division against finite differences and cover reverse subtraction, shared graph nodes and large tanh inputs. The verification script runs Python cells sequentially with a noninteractive plotting backend and updates saved text outputs. It does not verify Jupyter UI behaviour or Graphviz rendering.

Backprop uses scaled random weights and zero biases to reduce initial tanh saturation. Training is seeded and capped at 2,000 steps. Name sampling is capped at 40 characters per sample. Gradients still need to be reset before repeated backward passes. These are small learning demonstrations, not general-purpose ML libraries or evidence that the MLP is complete.

## Attribution

- [micrograd](https://github.com/karpathy/micrograd): autograd approach and copied Graphviz helper.
- [makemore](https://github.com/karpathy/makemore): character-model tutorial and names.txt dataset, verified against the upstream file.
- [Neural Networks: Zero to Hero](https://github.com/karpathy/nn-zero-to-hero): tutorial series.

Original MIT notices are retained in third-party/. The tutorial-derived implementation is credited to Andrej Karpathy; this repository records the learning work and subsequent corrections rather than claiming an original algorithm.
