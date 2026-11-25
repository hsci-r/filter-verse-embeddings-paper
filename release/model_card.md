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
model-index:
  - name: FILTER verse embeddings
---

# FILTER verse embeddings

The selected model from **Leveraging LLM-Generated Translations to Guide Verse Representation Learning in Finnic Oral Poetry**, by Lidia Pivovarova, Tinka Rainio and Eetu Mäkelä. Accepted to **Findings of EACL 2027**; an ACL Anthology identifier has not yet been assigned.

[Code and reproduction instructions](https://github.com/hsci-r/filter-verse-embeddings-paper).

## Model and training

The encoder returns 768-dimensional verse embeddings. It is a custom 12-layer pre-norm Transformer with hidden size 768, 12 attention heads, feed-forward size 3072, a final layer normalization, and a 128-token limit. It uses the supplied verse-balanced 32k SentencePiece tokenizer. The released checkpoint is from epoch 200, with training loss weights 0.45 verse alignment, 0.45 word alignment and 0.10 masked language modelling. Training dropout was 0.5.

English sentence targets come from `sentence-transformers/all-mpnet-base-v2`; word targets come from `answerdotai/ModernBERT-base`. Translations and word analyses were generated with DeepSeek 3. The exact upstream model revisions were not retained in the saved training configuration.

The paper dataset has 4,311,544 verse rows from 291,227 poems. The frozen training/development/test partitions have 232,793 / 29,109 / 29,325 poems. This release contains the trained encoder, its learned verse projection, the word projection when present, the tied MLM head, and the matching tokenizer. Optimizer state is omitted.

## Usage

Install the package from the accompanying code repository:

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

For offline use, pass a downloaded model directory. For a fixed release, supply `revision` with the Hub commit hash. A minimal installable copy of the loader is also included in this model repository; `pip install .` from a local snapshot installs it.

This model uses its own loader. Loading weights into a stock BERT architecture or changing the pooling rule changes the representations. Isolated encoding averages all non-padding tokens, including BOS/EOS. Contextual encoding packs consecutive verses from one poem into non-overlapping windows and averages only each verse's content tokens. Both apply the learned verse projection and L2 normalization. A single overlong verse is truncated to fit the window. An empty contextual verse produces a zero vector.

## Evaluation and use

On the frozen formulaic-verse retrieval benchmark, the selected model has P@1 **0.4091714003** and UAP **0.5553479680**. The benchmark has 43,483 queries and 145,150 directed positive edges, defined by English-translation similarity strictly above 0.99. Intrinsic masked-language-model loss is **1.7873176475**, with evaluation seed 13 and batch size 256. Full settings and saved results are in the code repository.

The intended use is research on verse similarity and Finnic oral poetry. Evaluation covers the paper's corpus and tasks; performance on other languages, genres and general sentence embedding tasks has not been established. English teacher representations and generated translations can carry errors into the learned representations.

## Licence and citation

Model weights and tokenizer: **CC BY 4.0** (`LICENSE`). Give appropriate credit, link to the licence and indicate changes. Loader code: **Apache-2.0** (`LICENSE_CODE`). Please cite the paper when using the model or tokenizer; the requested citation is in `CITATION.cff`.

`manifest.json` and `SHA256SUMS` identify the package files. `validation.json` records comparison with the original checkpoint and exporter. Zenodo archive links will be added after the deposits are published.
