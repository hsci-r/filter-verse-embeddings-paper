---
license: cc-by-4.0
library_name: pytorch
tags:
  - sentence-similarity
  - verse-embeddings
  - oral-poetry
  - finnish
  - estonian
  - custom-code
language:
  - fi
  - et
pipeline_tag: sentence-similarity
---

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

```bash
git clone https://github.com/hsci-r/filter-verse-embeddings-paper.git
pip install ./filter-verse-embeddings-paper
```

```python
from filter_verse_embeddings import FilterVerseEncoder

model = FilterVerseEncoder.from_pretrained(
    "University-of-Helsinki/filter-verse-embeddings", device="cpu"
)
verses = ["Metsä tuulessa laulaa.", "Joki kantaa illan valon."]
isolated = model.encode_verses(verses)
contextual = model.encode_poem(verses)
# Both arrays: float32, shape (2, 768).
```

Pass a downloaded model directory for offline use. Supply a Hugging Face `revision` to select a fixed release.

## Model

The model produces 768-dimensional embeddings using a 12-layer Transformer and a verse-balanced 32k SentencePiece tokenizer. It accepts up to 128 tokens per window. The selected checkpoint is from epoch 200, trained with loss weights 0.45 verse alignment, 0.45 word alignment and 0.10 masked language modelling.

English targets use [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) and [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base). Translations and word analyses were generated with `unsloth/DeepSeek-R1-BF16` from the [FILTER corpus](https://github.com/hsci-r/filter-data).

Isolated encoding pools each verse separately, including its boundary tokens. Contextual encoding packs consecutive verses from a poem and pools each verse's content tokens. Both apply the learned projection and L2 normalization. An overlong verse is truncated; an empty contextual verse returns a zero vector.

## Evaluation

On the paper's retrieval benchmark: **P@1 0.4092**, **UAP 0.5553**. Masked-language-model loss: **1.7873**. See [settings and saved results](https://github.com/hsci-r/filter-verse-embeddings-paper/blob/main/docs/reproduce.md).

The model is intended for research on Finnic oral poetry. Generated translations and English teacher representations can introduce errors; evaluation covers the paper's corpus and tasks.

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
