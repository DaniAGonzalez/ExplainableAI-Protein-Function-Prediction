# Protein Function Prediction with ESM-2: From Frozen Embeddings to Fine-Tuning

Predicting protein molecular functions from amino acid sequences using ESM-2, a protein language model pre-trained on UniRef50. This project compares two transfer learning strategies — frozen embeddings with a full explainability analysis, and end-to-end fine-tuning — and quantifies how much task-specific adaptation of a foundation model actually buys on this class of task.

## Why this exists

This is a minimum viable pipeline, built as a feasibility check before committing full compute to a downstream application that is not public. The question it answers is narrow and practical: do frozen ESM-2 embeddings already carry the signal, or does fine-tuning earn its cost? Running that comparison small, on public benchmarks, is cheaper than discovering the answer halfway through a full training budget.

Everything here runs on public data — Gene Ontology annotations over the UniProt human proteome. No proprietary data, models, or task definitions appear in this repository.

## Scope

A feasibility study, not a production pipeline. Two strategies on a bounded public benchmark, sized to answer one question about compute allocation. It does not include data engineering at scale, hyperparameter search, deployment, or the downstream task the comparison was run for.

## Project structure

```
├── 01_frozen_embeddings_interpretability.ipynb   # Frozen ESM-2 + full XAI pipeline
├── 02_finetuning.ipynb                           # Fine-tuned ESM-2 + performance comparison
├── figures/
└── README.md
```

## Motivation

Millions of sequenced proteins lack functional characterization, and homology-based approaches fail for proteins without close relatives. Protein language models like ESM-2 capture evolutionary and biochemical patterns from sequence alone, but a design question remains: **should these models be used as fixed feature extractors, or should their weights be adapted for the specific task?**

This project answers that question in order — first establishing a frozen baseline with deep interpretability analysis, then measuring what fine-tuning adds on top of it.

## Part 1: Frozen embeddings + interpretability

**Notebook:** `01_frozen_embeddings_interpretability.ipynb`

ESM-2 is used as a frozen feature extractor: sequences are encoded into fixed 640-dimensional representations, and a lightweight classifier is trained on top for multi-label Gene Ontology molecular function prediction.

The central contribution of this part is interpretability — understanding *why* predictions are made.

### Predictive performance (frozen)

| Metric | Value |
|--------|-------|
| Column-wise PR-AUC (per GO term) | 0.131 (8.4× over random) |
| Row-wise PR-AUC (per protein) | 0.349 (23.0× over random) |
| Best-performing class (olfactory receptors) | 0.996 |
| Protein kinases | 0.892 |

### Explainability analysis

Three complementary techniques applied to 40 systematically selected proteins across 10 molecular function categories:

**Attention visualization.** Well-predicted proteins show focused attention at specific residue positions (structural landmarks, domain boundaries), while poorly predicted proteins show dispersed attention. Attention identifies computational landmarks rather than directly marking functional residues.

**Integrated gradients.** Function-specific embedding dimensions emerge without explicit supervision: kinases rely on dimension 493, receptors on dimensions 532 and 565, transcription factors on dimensions 350 and 385. Dimension 553 acts as a general functional-protein signal across all families.

**In silico mutagenesis.** Single and combinatorial mutations cause minimal prediction changes (<5%), consistent with evolutionary robustness. Deletion scanning reveals critical regions: deleting the kinase activation loop drops the prediction by 10.7%, while receptors remain robust, consistent with architectural redundancy. This supports the reading that the model keys on genuine protein biology rather than dataset artifacts.

## Part 2: Fine-tuning

**Notebook:** `02_finetuning.ipynb`

The same ESM-2 model is fine-tuned end to end: the last 10 of 30 transformer layers are unfrozen and trained jointly with the classification head, allowing the encoder to adapt its internal representations for molecular function prediction.

### Fine-tuning strategy

| Parameter | Value |
|-----------|-------|
| Frozen layers | First 20 of 30 (embedding + early layers) |
| Trainable parameters | 50.1M (33.7% of total) |
| ESM-2 learning rate | 2e-5 |
| Classifier learning rate | 1e-3 |
| Epochs | 5 |
| Training time | ~10 min on A100 |

### Results: frozen vs fine-tuned

| Metric | Frozen | Fine-tuned | Change |
|--------|--------|------------|--------|
| Column-wise PR-AUC | 0.131 | 0.472 | +260% |
| Row-wise PR-AUC | 0.349 | 0.607 | +74% |
| Test loss | 0.0737 | 0.0541 | −27% |

### Function-specific comparison

| Function | Frozen | Fine-tuned | Change |
|----------|--------|------------|--------|
| Olfactory receptor | 0.996 | 1.000 | +0.4% |
| GPCR activity | 0.991 | 0.954 | −3.7% |
| Protein kinase | 0.892 | 0.954 | +7.0% |
| DNA-binding TF | 0.903 | 0.974 | +7.9% |
| Serine peptidase | 0.813 | 0.913 | +12.3% |
| Kinase activity | 0.650 | 0.989 | +52.1% |

**Key result.** Fine-tuning helps most where frozen embeddings were weakest, which is consistent with the encoder learning task-specific features that complement the general evolutionary patterns from pre-training. Classes already near ceiling (olfactory receptors, GPCR activity) show minimal change, and GPCR activity degrades slightly — reported here rather than smoothed over.

## Dataset

| Property | Value |
|----------|-------|
| Source | Gene Ontology Consortium + UniProt human proteome |
| Proteins | ~7,300–8,700 (varies by GO release) |
| GO terms | 103–202 molecular function terms |
| Max sequence length | 500 amino acids |
| Split | 60% train / 20% valid / 20% test |

Data is not distributed in this repository. All loading, preprocessing, and embedding generation is implemented in code; public data is downloaded directly from the GO Consortium and UniProt inside the notebooks.

## Methods summary

### Modeling pipeline

- Feature extraction / fine-tuning: ESM-2 (`facebook/esm2_t30_150M_UR50D`, 150M parameters)
- Mean pooling → 640-dimensional protein embeddings
- Classifier: 640 → 512 → 256 → n_labels (batch normalization, ReLU, dropout 0.3, sigmoid)
- Binary cross-entropy loss; early stopping (frozen) / linear warmup scheduler (fine-tuning)

### Interpretability techniques (frozen)

- Attention visualization (averaged across 30 layers × 20 heads)
- Integrated gradients (Captum, zero-embedding baseline)
- In silico mutagenesis (alanine scanning, combinatorial, deletion scanning)

## Quickstart

```bash
pip install torch transformers captum scikit-learn pandas numpy matplotlib seaborn biopython obonet
```

Run `01_frozen_embeddings_interpretability.ipynb` first — it downloads the data and produces the embeddings and baseline the second notebook compares against. `02_finetuning.ipynb` needs a GPU; the frozen notebook runs on CPU or Apple Silicon.

## Tested on

- Python 3.11+
- PyTorch 2.0+
- transformers 4.35+
- macOS (Apple Silicon, MPS), Linux (CUDA, A100)

## Future work

- Apply the full XAI pipeline (attention, integrated gradients, in silico mutagenesis) to the fine-tuned model and compare how fine-tuning changes interpretability patterns

## Citation

```bibtex
@misc{gonzalez2026esm2,
  title={Protein Function Prediction with ESM-2: From Frozen Embeddings to Fine-Tuning},
  author={Gonzalez, Daniela Alejandra},
  year={2026},
  institution={Northeastern University}
}
```

## License

Code is released under the MIT License.
Data usage is subject to Gene Ontology Consortium and UniProt terms.

## Contact

Daniela Alejandra Gonzalez

GitHub: [github.com/DaniAGonzalez](https://github.com/DaniAGonzalez)
