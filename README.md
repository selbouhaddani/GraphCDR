# GraphCDR
Source code and data for "GraphCDR: A graph neural network method with contrastive learning for cancer drug response prediction"

![Framework of GraphCDR](https://github.com/liuxuan666/GraphCDR/blob/main/Framework.png)  

# Requirements
* Python >= 3.8
* PyTorch >= 2.0 (the export workflow uses `torch.load(..., weights_only=True)`)
* PyTorch Geometry >= 1.6
* torch-scatter and torch-sparse compatible with your PyTorch installation
* NumPy, pandas, SciPy, and scikit-learn
* hickle >= 3.4
* DeepChem >= 2.4
* RDkit >= 2020.09

# Usage
* please unzip the file: data/Drug/drug_graph_feat.zip first.
* From the repository root, run `python prog/graphCDR.py <parameters>` for GDSC.
* Run `python prog/graphCDR-ccle.py <parameters>` for CCLE. This additionally needs DeepChem and RDKit.
* Data paths are resolved from the repository, so either script can be run from any working directory.

# Saved research objects

Both scripts save the selected model and its predictions automatically. For a quick run with your activated environment:

```bash
python prog/graphCDR.py --epoch 1 --seed 0 --split-seed 666 --output-dir runs/gdsc-check
```

Omit `--output-dir` to create a new directory under `runs/<dataset>/` with a UTC timestamp and the training seed. To reuse an existing GraphCDR run directory, add `--overwrite`:

```bash
python prog/graphCDR.py --epoch 1 --seed 0 --split-seed 666 --output-dir runs/gdsc-check --overwrite
```

This starts a fresh run and replaces the previous generated artifacts and `source/` snapshot; unrelated files such as notes are retained. Nonempty directories must contain a valid GraphCDR `settings.json` to be overwritten. Without the flag, existing directories are protected. The usual epoch defaults are 350 for GDSC and 200 for CCLE.

| File | Contents |
| --- | --- |
| `research_objects.npz` | Ordered node IDs, complete final embeddings, logits, probabilities, labels, split masks, training graph, selected epoch, and classification threshold |
| `predictions.csv` | All cell-line/drug pairs with indices, identifiers, label, split, logit, probability, and predicted binary label |
| `best.pt` | Selected model and optimizer state, architecture settings, ordered IDs, labels, split masks, training edges, training RNG state, and validation metrics |
| `settings.json` | Arguments, seeds, package versions, feature names, input/source SHA-256 hashes, and method details |
| `history.csv` | Training loss and validation metrics for every epoch |
| `metrics.json` | Train, validation, and held-out test metrics for the selected checkpoint |
| `source/` | Copy of the Python implementation used for this run |

The response split is stratified by sensitivity label: approximately 80% training, 10% validation, and 10% test, controlled by `--validation-fraction`, `--test-fraction`, and `--split-seed`. This is a transductive split of observed **pairs**, not a holdout of entire cell lines or drugs. Only training responses enter the response graph and response loss. The first epoch with the highest validation AUC is selected and reloaded before the final export; the test set is evaluated only after selection. The classification threshold maximizes validation F1 and is also used for train/test F1 and accuracy. AUPR is the trapezoidal area under the precision-recall curve.

`--seed` controls model initialization independently of the response split. Keep the split seed and fractions fixed when comparing models or training seeds. For example:

```bash
python prog/graphCDR.py --seed 1 --split-seed 666 --output-dir runs/gdsc-seed1
python prog/graphCDR.py --seed 2 --split-seed 666 --output-dir runs/gdsc-seed2
```

The NPZ archive can be loaded without pickle. Matrix rows follow `cell_line_ids` and columns follow `drug_ids` exactly; IDs are strings (cell IDs from the source response table and PubChem drug IDs). Embedding rows use those same orders. `labels` contains 1 for sensitivity, 0 for resistance, and NaN for unobserved responses. Predictions are saved for **all** pairs, but unobserved responses are excluded from metrics. `split` codes are 0 = unobserved, 1 = train, 2 = validation, 3 = test. Boolean `observed_mask`, `train_mask`, `validation_mask`, and `test_mask` are also saved. Checkpoint labels use a zero placeholder for unobserved pairs and must be read with the observed mask. `train_edges` contains directed `(source, destination, response)` rows, where response is +1/-1 and graph nodes are all cell lines followed by all drugs. Epoch numbers are one-based.

For an individual pair, reconstruct the coordinate contributions without storing a cell-by-drug-by-coordinate tensor:

```python
import numpy as np

with np.load("runs/gdsc-check/research_objects.npz", allow_pickle=False) as saved:
    i, j = 0, 0  # Select by the saved IDs, not by input-file row order.
    cell_id, drug_id = saved["cell_line_ids"][i], saved["drug_ids"][j]
    contributions = saved["cell_embeddings"][i] * saved["drug_embeddings"][j]
    logit = contributions.sum()
    probability = 1 / (1 + np.exp(-logit))
    assert np.isclose(logit, saved["logits"][i, j], rtol=1e-5, atol=1e-5)
    assert np.isclose(probability, saved["probabilities"][i, j])
```

The first `hidden_channels` coordinates come from the response graph; the final 10 come from the projected cell/drug features. These are the complete embeddings used in prediction, including the coordinates missing from `pos_z` alone. The sum equals the log odds mathematically, subject to floating-point rounding. Coordinates are unsupervised and have no predefined pathway meaning; contributions explain the prediction and do not establish causal pathway effects. Individual embedding axes can rotate or permute between runs, so coordinate indices alone do not establish biological reproducibility.

To reconstruct the selected model in an environment with this run's source:

```python
import sys
sys.path.insert(0, "runs/gdsc-check/source")
import torch
from research import build_model

checkpoint = torch.load("runs/gdsc-check/best.pt", map_location="cpu", weights_only=True)
model = build_model(checkpoint["model_config"])
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
model.eval()
```


Run the research export checks with:

```bash
python -m unittest discover -s tests -v
```
  
# Case study
As GDSC database only measured IC50 of part cell line and drug pairs. We applied GraphCDR to predicted the missing types of responses. The predicted results can be find at data/Case study (missing pairs).xlsx
