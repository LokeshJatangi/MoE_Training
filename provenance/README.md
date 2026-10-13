# Provenance and consolidation

The external original outputs are the backup, including original Git histories and checkpoint files. This is a fresh local submission repository.

- file_manifest.json accounts for each original non-cache file using base64-encoded exact original relative paths, original SHA-256, submission location, and disposition. Sanitized display paths avoid legacy branding.
- source_verification.json compares environment-recorded original source hashes with located original snapshots, then records the submitted source hash. original_verified=false is an unresolved historical mismatch, not a successful audit.
- deduplication.json lists relative aliases and their canonical files. Path-based Python readers and notebook zip writers dereference these links normally.
- validation.json records verification performed during consolidation. Numeric JSON and JSONL measurements are compared with originals after renaming keys; no training is rerun.

Submission environment source hashes describe the sanitized snapshots. Original digests are retained separately; original metadata must not be presented as an audit of changed bytes. Previously derived checkpoint reports are retained historical evidence. A metadata-only audit cannot independently validate checkpoint state.
