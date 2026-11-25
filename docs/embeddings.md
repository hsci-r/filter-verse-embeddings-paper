# Precomputed verse embeddings

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

Download [isolated.parquet](https://zenodo.org/records/24536159/files/isolated.parquet?download=1), [contextual.parquet](https://zenodo.org/records/24536159/files/contextual.parquet?download=1) and [row_alignment.parquet](https://zenodo.org/records/24536159/files/row_alignment.parquet?download=1).

Each embedding table has **4,275,969 rows**:

| Column | Meaning |
| --- | --- |
| `p_id` | Poem ID |
| `pos` | Verse position in the poem |
| `v_id` | Verse-text ID; the same text can occur more than once |
| `embedding` | 768 float32 values |

The tables cover the same occurrences in different row orders. Join by `(p_id, pos, v_id)`. The alignment file also maps their zero-based `isolated_row` and `contextual_row` indices. Original text and metadata are available in [FILTER data](https://github.com/hsci-r/filter-data); `metadata/poems.parquet` in the [paper data archive](https://zenodo.org/records/24536154) maps poem IDs to archival identifiers.

Isolated embeddings represent each verse independently. Contextual embeddings represent a verse within a packed window of its poem. Both use the released model's learned projection and L2 normalization.

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
