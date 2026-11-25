# Release resources

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

The four Zenodo records preserve the code, model and tokenizer, paper data, and precomputed embeddings. Hugging Face provides the model for direct loading. The original corpus is maintained in [FILTER data](https://github.com/hsci-r/filter-data).

`prepare_zenodo.py` builds the archives. `upload_zenodo.py` uploads and publishes them; `publish_hf.py` publishes a verified model package. Run large transfers and archive preparation in a compute allocation. Credentials are read from environment variables.

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
