# Permanent release archives

GitHub is the working code repository. Hugging Face distributes the selected model and tokenizer. Zenodo preserves the versioned code, model, data and embeddings as downloadable files.

The release has four archive groups:

| Archive | Contents |
| --- | --- |
| Code | Complete source tree from a Git commit: source, configurations, saved results, figures, documentation, citation and licences |
| Model | Weights, tokenizer, native loader, model card, licences, inference settings and validation records |
| Data | Translation and word-analysis Parquets, exact paper snapshot, frozen splits, retrieval positives, tokenizers and supporting corpus tables |
| Embeddings | Best-model isolated and contextual Parquets, their schemas, row coverage and checksums |

Record the code commit, Hugging Face revision, archive SHA-256 values and Zenodo record identifiers together. Add the eventual paper DOI/Anthology URL once assigned. A GitHub URL alone does not archive code or model bytes on Zenodo.

`download_artifacts.py` mirrors the selected public Allas files without changing their bytes. Run substantial transfers in a compute allocation. It checks the recorded source size and ETag, computes SHA-256, and refuses to replace an unverified target. The isolated/contextual files total about 23 GB; all 13 selected public files total 23,501,703,277 bytes.

`publish_hf.py` verifies a completed model package and uploads it using the Hugging Face Python client. It uses the locally configured HF login or `HF_TOKEN`; credentials are never stored in this repository. Run large uploads in a compute allocation.

Before publishing Zenodo records, finish artifact validation and supply the applicable licences for corpus data and generated translations. Model/tokenizer licence: CC BY 4.0. Code licence: Apache-2.0. Do not apply the code licence to all data by default.
