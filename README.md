# FILTER verse embeddings

Resources for **Leveraging LLM-Generated Translations to Guide Verse Representation Learning in Finnic Oral Poetry**, by Lidia Pivovarova, Tinka Rainio and Eetu Mäkelä. Accepted to **Findings of EACL 2027**.

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

## Use the model

Install the package in an environment with PyTorch:

```bash
pip install .
```

```python
from filter_verse_embeddings import FilterVerseEncoder

model = FilterVerseEncoder.from_pretrained(
    "University-of-Helsinki/filter-verse-embeddings", device="cpu"
)
verses = ["Metsä tuulessa laulaa.", "Joki kantaa illan valon."]
isolated = model.encode_verses(verses)
contextual = model.encode_poem(verses)
# Both arrays have shape (2, 768).
```

Use a local model directory for offline inference. The package provides the model's custom architecture and pooling.

## Train the model

Training combines masked language modelling with alignment to English verse and word representations. The selected configuration trains from scratch for 200 epochs with batch size 32 and learning rate 2.5e-4. All [20 experiment configurations](https://github.com/hsci-r/filter-verse-embeddings-paper/tree/main/configs/experiments) are included.

Follow the [data instructions](https://github.com/hsci-r/filter-verse-embeddings-paper/blob/main/docs/data.md) to place the paper input and tokenizer under `artifacts/data/`. Word-supervised configurations also require `artifacts/data/word_analysis_training.parquet` with the original training IDs and row order; that table is unavailable in this release. With the required inputs, run from the repository root in a compute allocation:

```bash
pip install '.[reproduction]'
python -m filter_verse_embeddings.training \
  --config configs/experiments/verse_balanced_v0p45_w0p45_mlm0p10.yaml
```

The trainer computes and caches English teacher embeddings automatically. Checkpoints are saved under `artifacts/checkpoints/verse_balanced_v0p45_w0p45_mlm0p10/`. See [training and evaluation details](https://github.com/hsci-r/filter-verse-embeddings-paper/blob/main/docs/reproduce.md) for input requirements, settings and reproducibility limits.

## Data and experiments

The corpus comes from [FILTER data](https://github.com/hsci-r/filter-data). Translations and word analyses were generated with **DeepSeek 3**. The paper uses 4,311,544 verse rows from 291,227 poems. Each precomputed embedding table contains 4,275,969 verse occurrences.

See [data and downloads](https://github.com/hsci-r/filter-verse-embeddings-paper/blob/main/docs/data.md), [embedding usage](https://github.com/hsci-r/filter-verse-embeddings-paper/blob/main/docs/embeddings.md), and [reproduction instructions](https://github.com/hsci-r/filter-verse-embeddings-paper/blob/main/docs/reproduce.md). The repository includes 20 experiment configurations, frozen splits, saved results and figures.

## Licence

Code: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0). Model weights, tokenizer, generated translations, word analyses and embeddings: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Original corpus text and metadata retain the terms described in [FILTER data](https://github.com/hsci-r/filter-data#license).

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
