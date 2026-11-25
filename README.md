# FILTER verse embeddings

Code and experiment configurations for **Leveraging LLM-Generated Translations to Guide Verse Representation Learning in Finnic Oral Poetry**, by Lidia Pivovarova, Tinka Rainio and Eetu Mäkelä. Accepted for publication in **Findings of the Association for Computational Linguistics: EACL 2027**. The ACL Anthology URL will be added after publication.

The selected model learns 768-dimensional verse representations using masked language modelling and alignment to English teacher embeddings. It uses a verse-balanced 32k SentencePiece tokenizer, loss weights 0.45 verse / 0.45 word / 0.10 MLM, and the checkpoint after 200 epochs.

## Use the model

Install this package in an environment with PyTorch:

```bash
pip install .
```

The model and matching tokenizer are published together at [University-of-Helsinki/filter-verse-embeddings](https://huggingface.co/University-of-Helsinki/filter-verse-embeddings). Once the model files have been uploaded:

```python
from filter_verse_embeddings import FilterVerseEncoder

model = FilterVerseEncoder.from_pretrained(
    "University-of-Helsinki/filter-verse-embeddings",
    device="cpu",
)
verses = ["Metsä tuulessa laulaa.", "Joki kantaa illan valon."]
isolated = model.encode_verses(verses)
contextual = model.encode_poem(verses)
print(isolated.shape, contextual.shape)  # (2, 768), (2, 768)
```

Use a local model directory in place of the Hub ID for offline inference. Pass an immutable Hub `revision` to reproduce a particular release. The encoder has a custom architecture and loader; use this package to obtain the published projections and pooling behavior.

Isolated encoding pools all non-padding sequence tokens, including BOS/EOS. Contextual encoding pools only the tokens belonging to each verse. Contextual windows stay within a poem and have at most 128 tokens including special tokens. Long poems use consecutive non-overlapping windows; overlong individual verses are truncated. Output vectors are projected and L2-normalized. A poem vector can be formed by averaging its verse vectors.

## Reproduce the paper

See [docs/reproduce.md](docs/reproduce.md). The 20 final experiment configurations are in `configs/experiments/`, with their saved retrieval, masked-language-model and contextual/isolated classification results in `results/`. Frozen split lists are in `data/splits/`.

The exact cleaned paper data contains 291,227 poems and 4,311,544 rows. Existing public translation and embedding Parquets use canonical corpus IDs and may have different coverage. The data and embedding manifests describe their respective row sets. Allas source URLs and recorded sizes are listed in [data/allas_sources.json](data/allas_sources.json); Zenodo version links will be added after the deposits are created.

## Licence and citation

The code is licensed under **Apache-2.0**. The published best-model weights and matching tokenizer are licensed under **CC BY 4.0**, requiring appropriate attribution, a licence link and an indication of changes. Source corpus data and translations retain their applicable source licences; the code licence does not relicense them.

Please cite the paper when using the code, model, tokenizer, translations or embeddings. The requested citation is recorded in [CITATION.cff](CITATION.cff). Publication in the ACL Anthology is pending; no paper DOI or Anthology identifier is asserted here.
