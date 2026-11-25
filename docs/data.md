# Paper data

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

## Downloads

Download the following files from the [data archive](https://zenodo.org/records/24536154) into `artifacts/data/`:

- `verses_translated.parquet`: unsloth/DeepSeek-R1-BF16 verse translations.
- `word_analysis.parquet`: unsloth/DeepSeek-R1-BF16 word analyses and English translations, with 14,618,529 rows.
- `paper_verses.parquet`: the cleaned paper input, with 4,311,544 verse rows from 291,227 poems.
- `filter-verse-embeddings-data-support-v1.0.0.tar.gz`: splits, retrieval pairs, tokenizers, corpus metadata and cleaning records. Extract it into the same directory.

Original corpus text and metadata come from [FILTER data](https://github.com/hsci-r/filter-data).

## Splits and word IDs

| Split | Poems | Verse rows |
| --- | ---: | ---: |
| Training | 232,793 | 3,441,993 |
| Development | 29,109 | 438,303 |
| Test | 29,325 | 431,248 |

The paper input also retains 4,490 rows without split assignments. The word-analysis table uses `p_id`, `v_pos`, `w_pos` and `w_id`. The original training-order word table is unavailable, so exact word-supervised retraining cannot be reconstructed from this table alone. The released model and embeddings can be used directly. See [reproduction details](https://github.com/hsci-r/filter-verse-embeddings-paper/blob/main/docs/reproduce.md).

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
