# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.3] - 2026-10-06

This update adds buffer writing for `--archive` and adds a new `--datalog` argument for archiving classifications

### Added
 - `--datalog <path>` (`BS_DATALOG`): appends every document submitted for classification to a CSV file in the same
   `labels,content` format as `--archive`, with the predicted labels in the `labels` column.

### Fixed
 - Concurrent `PUSH /` requests could interleave rows in the `--archive` CSV or lose rows while the file was being created.

### Changed
 - `--archive` (and `--datalog`) rows are now written by a background thread through a buffered writer instead of
   opening the file on every request. If the disk falls behind, rows are dropped rather than slowing down requests.



## [1.0.2] - 2026-09-28

This update introduces improvements to accuracy and bug fixes

### Added
 - `--mml-global-training` (default `true`): in MML mode every document is also trained into the `"und"` model, making
   it a global fallback that has seen every label in every language. Statistics and `--max-docs` count each document once.
 - `--mml-min-label-docs` (default `10`): in MML mode a language-specific model is only used for classification when every
   label has at least this many documents in it; otherwise the `"und"` model classifies the text.
 - `scripts/rebuild_from_archive.py` to de-duplicate and relabel an `--archive` CSV and replay it into a fresh model.

### Fixed
 - MML classified every text of a language with a single-label model as that label with probability `1.0` (for example
   any Hindi text as `MALICIOUS` when only Hindi spam had been trained). Lingua's confidence is relative and always `1.0`
   for the top language, so short or ambiguous text was routinely routed to such sparse models.



## [1.0.1] - 2026-08-16

This update introduces security fixes

### Fixed
 - Made HTTP method routing case-sensitive and added a configurable request read timeout (`--request-read-timeout-ms`) to close stalled connections.
 - Prevented label filename collisions and oversized filename failures with injective escaping and bounded hashed filenames.
 - Capped persistable token length and labels per document; label-pair tracking now runs only when label-chain mode is enabled.


## [1.0.0] - 2026-06-19

Initial release of BayesianServer