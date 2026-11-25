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

`prepare_zenodo.py` validates the frozen artifact hashes, archives a specific Git commit and the published model package, and creates `upload_plan.json`. The data deposit exposes translations, word analyses and the exact paper snapshot as direct Parquet downloads, with the remaining split, benchmark, tokenizer and provenance files in a support archive. The embedding deposit exposes both original Parquets and the row-alignment map directly.

The four metadata templates in `release/zenodo/` name all three paper authors and preserve the accepted EACL 2027 citation without inventing a paper DOI. The code licence is Apache-2.0. Model/tokenizer, generated translations, analyses and embeddings use CC BY 4.0; original corpus text and upstream metadata retain their source terms.

`upload_zenodo.py upload_plan.json` uses `ZENODO_TOKEN` from the environment. It verifies local hashes, creates four records, reserves their DOIs and cross-links them. All metadata must be accepted before large uploads begin. Each uploaded file is checked against Zenodo's returned MD5 and size. The script publishes all four records after every planned file has passed verification. It records draft IDs and uploaded checksums in `zenodo_state.json`, allowing a failed upload to resume without creating duplicate records. `--draft-only` is available for archival workflows that need unpublished drafts.

Before creating records, the uploader establishes HTTPS with a read-only request. Connection failures on read-only requests receive at most three attempts. Record creation, streamed uploads and publication are each attempted once; an ambiguous write timeout requires checking the existing Zenodo drafts and saved state before resubmitting.

Large transfers belong in a compute allocation. The token needs `deposit:write` and `deposit:actions`; enter it in your own terminal. `pip install '.[release]'` installs the upload helper's Requests dependency if needed. DOI links can be added to GitHub and the HF model card after publication. The code archive identifies its immutable Git commit even when documentation receives later DOI updates.
