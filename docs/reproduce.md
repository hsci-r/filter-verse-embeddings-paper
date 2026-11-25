# Reproducing the paper experiments

Use the frozen poem split assignments and benchmark rather than drawing new partitions or mining new positives. Install reproduction dependencies with `pip install '.[reproduction]'`. Downloads, preprocessing, model inference, training and evaluation should run in a suitable compute allocation; the reference runs used Slurm.

## Artifacts and paths

Place the data deposit under `artifacts/data/` and the selected model under `artifacts/model/`. The expected exact paper input is `artifacts/data/paper_verses.parquet`; tokenizers are `artifacts/data/tokenizers/{standard,verse_balanced}/tokenizer.model`. Keep the public translation and word-analysis Parquets alongside this exact paper snapshot. The original ordered word table expected by training is `artifacts/data/word_analysis_training.parquet`.

Before adapting the public word-analysis table, verify its schema and row-order mapping to the training table. The word teacher contextualizes consecutive rows in stitched chunks, so reordering rows can change supervision. If the schemas differ, use the recorded adapter supplied with the verified data package. Do not silently replace inputs or use a new translation run.

The selected public word-analysis Parquet has 14,618,529 rows with corpus coordinates `p_id`, `v_pos`, `w_pos`, and `w_id`. Its English translations and analyses are preserved unchanged. It lacks the training reader's string `poem_id`, `word_id`, `word_position` and `original_form` columns. The identity mapping and original training-row order still need verification before the retraining input can be declared complete.

The supplied split manifest records 232,793 / 29,109 / 29,325 poems and 3,441,993 / 438,303 / 431,248 verse rows for train / development / test. Keep 4,490 retained unassigned translation rows, including their nullable positions. The cleaning rule excludes a poem only when more than 50 percent of its translation rows lack source positions.

The public isolated and contextual embedding tables each contain 4,275,969 rows, with columns `p_id`, `pos`, `v_id` and `embedding`. The embedding column stores lists of 768 float32 values. These corpus occurrence coordinates differ from the paper's string IDs. Join `p_id` to `metadata/poems.parquet` (`p_id` to archival `nro`); `v_id` identifies a verse text and can recur in multiple occurrences. Compare the tables using their supplied occurrence identifiers. The release audit records identity coverage and row order separately; their positional alignment is still under review.

## Training

Run from the repository root:

```bash
python -m filter_verse_embeddings.training \
  --config configs/experiments/verse_balanced_v0p45_w0p45_mlm0p10.yaml
```

`configs/experiments.tsv` lists all 15 main configurations and five FinEst-tokenizer initialization ablations. They retain the saved numerical training settings: seed 42, 200 epochs, learning rate 2.5e-4, dropout 0.5, batch size 32, sequence length 128, temperature 0.05, word loss scale 10, weight decay 0.01 and gradient clipping 0.25. The sentence teacher is `sentence-transformers/all-mpnet-base-v2`; the word teacher is `answerdotai/ModernBERT-base`.

Teacher targets are cached under `artifacts/cache/`. The original upstream revision identifiers and complete training environment were not recorded in the saved YAML configurations. A release validation environment is supplied separately after verification; it is not presented as the original training environment. Hardware and library differences may affect retraining results.

## Export and evaluation

Export test embeddings from a recreated training run with:

```bash
python -m filter_verse_embeddings.export_evaluation \
  --config configs/experiments/verse_balanced_v0p45_w0p45_mlm0p10.yaml \
  --checkpoint artifacts/checkpoints/verse_balanced_v0p45_w0p45_mlm0p10/epoch_200.pt
```

The retrieval evaluator consumes a run directory containing `embeddings/test_filter.npy` and `metadata/test_metadata.parquet`:

```bash
python -m filter_verse_embeddings.evaluation.retrieval \
  --model-run artifacts/runs/verse_balanced_v0p45_w0p45_mlm0p10 \
  --benchmark-pairs artifacts/data/benchmarks/retrieval/positive_pairs.parquet \
  --output-path artifacts/results/retrieval.json
```

The final benchmark uses English similarity strictly above 0.99, with 43,483 queries and 145,150 directed positive edges. It ranks original-language candidates and masks the query itself. Keep the saved positive pairs and candidate ordering fixed. The best saved result is P@1 0.4091714003 and UAP 0.5553479680.

For intrinsic MLM evaluation use `python -m filter_verse_embeddings.evaluation.mlm --help` with test split, seed 13, batch size 256 and packed poem windows. The selected model's saved loss is 1.7873176475 over 452,947 masked tokens. Changing batching can change the stochastic masks even with the same seed.

For poem-type classification use `python -m filter_verse_embeddings.evaluation.poem_types --help`. Set minimum train support to 51 and exclude the `erab_orig` prefix. The saved task contains 26 labels and 112,706 / 14,127 / 14,267 labelled poems. Public type-table IDs and columns must be adapted explicitly to the archived task assignments.

## Tokenizers and figures

`tokenizers.train` and `tokenizers.balance` retain the unigram SentencePiece training and verse-balancing algorithms. Tokenizer retraining requires the original sampling settings and dialect/place tables; use the distributed tokenizer bytes for inference and model comparisons.

`figures.retrieval` and `figures.mlm` generate the numerical overview figures from saved result tables. The released PNG/PDF figures and per-model JSON results preserve the reported outputs. The qualitative heatmaps show `skvr07106440` versus `skvr07104320`; their original small numeric source matrices still need to be included with the final figure-reproduction recipe.
