# Reproducing the experiments

## Resources

| Resource | Link |
| --- | --- |
| Code and instructions | [GitHub](https://github.com/hsci-r/filter-verse-embeddings-paper) |
| Model and tokenizer | [Hugging Face](https://huggingface.co/University-of-Helsinki/filter-verse-embeddings) |
| Permanent code archive | [Zenodo](https://zenodo.org/records/24536150) |
| Permanent model archive | [Zenodo](https://zenodo.org/records/24536152) |
| Translations, word analyses and paper data | [Zenodo](https://zenodo.org/records/24536154) |
| Isolated and contextual embeddings | [Zenodo](https://zenodo.org/records/24536159) |
| Original corpus | [FILTER data](https://github.com/hsci-r/filter-data) |

## Setup and inputs

```bash
pip install '.[reproduction]'
```

Follow the [data download instructions](data.md) and place the model files from [Hugging Face](https://huggingface.co/University-of-Helsinki/filter-verse-embeddings) in `artifacts/model/`. Run downloads, preprocessing, inference, training and evaluation in a compute allocation.

Use the released split assignments and retrieval pairs. The cleaning rule excludes a poem when more than 50% of its translated rows lack source positions. The retained 4,490 unassigned rows keep their nullable positions.

## Reproducibility limits

The trained model, matching tokenizer, paper input, splits, retrieval pairs, precomputed embeddings and saved evaluation results are included. The original ordered word-supervision input, exact upstream model revisions and complete training environment are unavailable. The public word-analysis table uses different IDs and does not establish the original training-row order; consecutive word rows affect contextual teacher targets. Exact word-supervised retraining therefore requires information absent from this release.

The figure files are included. The original numeric matrices for the qualitative verse heatmaps are unavailable. The validation environment describes release checks, rather than the original training environment.

## Training

```bash
python -m filter_verse_embeddings.training \
  --config configs/experiments/verse_balanced_v0p45_w0p45_mlm0p10.yaml
```

`configs/experiments.tsv` lists 15 main configurations and five tokenizer-initialization ablations. Shared settings: seed 42, 200 epochs, learning rate 2.5e-4, dropout 0.5, batch size 32, sequence length 128, temperature 0.05, word-loss scale 10, weight decay 0.01 and gradient clipping 0.25. Teachers: `sentence-transformers/all-mpnet-base-v2` and `answerdotai/ModernBERT-base`. Tokenizers are supplied in the data archive; use those files for model comparisons.

## Export and retrieval

For a training run with the required inputs:

```bash
python -m filter_verse_embeddings.export_evaluation \
  --config configs/experiments/verse_balanced_v0p45_w0p45_mlm0p10.yaml \
  --checkpoint artifacts/checkpoints/verse_balanced_v0p45_w0p45_mlm0p10/epoch_200.pt

python -m filter_verse_embeddings.evaluation.retrieval \
  --model-run artifacts/runs/verse_balanced_v0p45_w0p45_mlm0p10 \
  --benchmark-pairs artifacts/data/benchmarks/retrieval/positive_pairs.parquet \
  --output-path artifacts/results/retrieval.json
```

The evaluator expects `embeddings/test_filter.npy` and `metadata/test_metadata.parquet` in the run directory. The benchmark contains 43,483 queries and 145,150 directed positive edges, defined by English-translation similarity strictly above 0.99. It ranks original-language candidates and masks the query itself. The selected model scores P@1 **0.4091714003** and UAP **0.5553479680**.

## Other evaluations and figures

- MLM: `python -m filter_verse_embeddings.evaluation.mlm --help`. Use the test split, seed 13, batch size 256 and packed poem windows. Saved loss: **1.7873176475**, over 452,947 masked tokens. Batching affects stochastic masks.
- Poem types: `python -m filter_verse_embeddings.evaluation.poem_types --help`. Minimum training support is 51; exclude `erab_orig`. The saved task has 26 labels and 112,706 / 14,127 / 14,267 labelled poems. Adapt public table IDs to the supplied task assignments.
- Embedding table joins: see [embedding usage](embeddings.md).
- Retrieval and MLM figures: use the `figures.retrieval` and `figures.mlm` modules with saved results. The qualitative heatmaps compare `skvr07106440` and `skvr07104320`.

## Citation

Please cite the paper when using these resources:

```bibtex
@inproceedings{pivovarova-etal-2027-leveraging,
  title = "Leveraging {LLM}-Generated Translations to Guide Verse Representation Learning in Finnic Oral Poetry",
  author = "Pivovarova, Lidia and Rainio, Tinka and M{\"a}kel{\"a}, Eetu",
  booktitle = "Findings of the Association for Computational Linguistics: {EACL} 2027",
  year = "2027",
  publisher = "Association for Computational Linguistics"
}
```
