# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.4] - Unreleased

This update adds corrections and a logistic regression model, and fixes the tokenizer for scripts with combining
marks and for Turkish

### Added
 - `--model-type logistic`: one-vs-rest logistic regression over hashed word and character features, retrained in the
   background (`--retrain-interval`) from a store of every learned text with its latest labels, with decision
   thresholds calibrated for a false positive rate (`--target-fpr`, default `0.05`). Options `--hash-bits`,
   `--regularization` and `--epochs`
 - `negative_label` of `POST /` (logistic model): the label of documents where nothing was detected (eg; `NORMAL`),
   another label is then only predicted when the probability of not being negative is above the negative label's
   calibrated threshold. Every retrain calibrates such a threshold for every label, reported as `negative_thresholds`
   by `GET /`
 - `--placeholders` replaces links, mentions, phone numbers, prices, numbers and emoji with placeholder words, and
   invisible characters with a marker, instead of removing them. The rules are generic, `--placeholder-rules` adds the
   rules of an application or replaces them
 - `--stop-words false` keeps every word
 - `--dedup`: learning a text again replaces its earlier labels (the latest label of a text wins)
 - `PATCH /` corrects the labels of learned texts. It takes the same body as `PUSH /`; every earlier training of each
   text is removed from the model and the text is learned once with the corrected labels, leaving the model exactly as
   if the text had only ever been learned with them. Corrections are archived like training rows
 - `--corrections <bool>` (`BS_CORRECTIONS`, default `true`) enables `PATCH /` and the training ledger, which records
   how every text was trained (a 64-bit hash of the text, the language model and the labels, about 100 bytes per
   distinct text) and is saved next to the model as `<model-path>.ledger`. A ledger written with different
   preprocessing options is ignored, texts learned without a ledger are only learned again when corrected
 - `NaiveBayesModel.untrain()` and `LanguageModelManager.untrain()` to remove learned documents
 - `scripts/rebuild_from_archive.py --latest-wins` resolves conflicting labels with the labels a document was trained
   with last, so replaying an archive keeps corrections
 - Document weights (logistic model): `weight` of `PUSH /` and `PATCH /` scales how much a document counts in
   training, documents weighing less than 1 don't calibrate the thresholds. Weights are stored with the documents
   (`corpus.bin` format 2, format 1 is still read) and archived in a third `weight` column
 - `DELETE /` forgets texts (logistic model, and Naive Bayes with `--corrections`), archived as rows without labels
 - `texts` of `POST /` classifies up to 1000 texts in one request
 - Implicit learning (`--implicit-delay`, `--implicit-weight`, `--implicit-capacity`): texts a classification
   predicted with only its `implicit_label` (eg; `NORMAL`) are learned with a low weight after a delay, unless they're
   learned, corrected or forgotten in the meantime, so the model sees ordinary traffic and not only reported content
 - `--retrain-threads`: the logistic model featurizes documents and trains its labels and its calibration model in
   parallel, with the same result as one thread. `--feature-cache` keeps document features between retrains
 - `scripts/rebuild_from_archive.py` reads and writes the weight column and applies forgotten texts
 - `--embedding-model` (logistic model): a sentence embedding model run with ONNX Runtime (eg; multilingual-e5-small)
   whose classifier is averaged with the hashed features, so spam is recognized in other words and languages
 - The data log records the time, the top label and probability, the negative label's probability and the model
   version of every classification (a data log with the old columns is moved aside)
 - `scripts/labelling_queue.py` builds a labelling queue of the texts the model was least sure about from the data log
   and pushes the labelled queue back, `scripts/evaluate.py` measures flagged ordinary content and caught spam for every
   `minimum_confidence` of a client like BayesianPlugin

### Removed
 - `NaiveBayesModel.calibrateThresholds()` and the per-label thresholds of the Naive Bayes model, which were never
   calibrated by the server (only by tests), so every label always used `--threshold`. Models are unaffected
 - `NaiveBayesModel.hasLabels()`, `ApiException.serviceUnavailable()` and `Utilities.evaluateMetric()`, which were unused

### Fixed
 - The training archive no longer drops rows when the disk can't keep up, requests wait for room instead, so a model
   rebuilt from it matches the model that wrote it
 - `GET /` of the logistic model no longer visits every document and weight on every request, its statistics are
   cached until the model changes
 - `--memory-limit` limits memory: a label was only weighed when it was loaded, so a label growing through learning
   kept the weight it had when loaded (nothing for a new label) and was never evicted. Learning and unlearning now
   change a label's counts atomically with its eviction, so changes are no longer lost when a label is evicted while
   it's learned, and weigh the label again
 - `NaiveBayesModel.evictLabel()` persists the label before removing it from memory instead of dropping its counts
 - A label evicted under `--memory-limit` before the model was saved for the first time is loaded back with its
   counts instead of empty, loading a label only requires the label index rather than a complete saved model
 - The logistic model honors `--dedup`: `PUSH /` without it adds the new labels of a known text to its labels instead
   of replacing them, as `PATCH /` does
 - Documents rejected because the learning queue is full are no longer written to the archive
 - Labels containing `;`, which separates labels in the archive, are rejected with `400`
 - `GET /` reports the `corrected` and `uncorrectable` counts of the learning queue
 - The tokenizer split words at combining marks, breaking Hindi, Bengali, Malayalam, Myanmar, Thai and other scripts
   into fragments or nothing, combining marks now stay in their word
 - Upper-case Turkish words produced different tokens than lower-case ones (`İ` became `i` and a combining dot, `ı`
   stayed dotless), both are folded to `i`. Models should be rebuilt, tokens of affected texts change

### Changed
 - Updated to Jackson 3 (`tools.jackson`), Netty 4.2.19 and Logback 1.6.5. Requests and responses are unchanged:
   properties keep their declaration order, `null` for a number field is still accepted and malformed JSON is still
   rejected with `400`
 - `docker-compose.yml` runs the recommended spam detection configuration: `logistic` model, `--placeholders true`,
   `--stop-words false`, `--dedup true`, `--hash-bits 16` and `--target-fpr 0.05`
 - The learning queue is split into one queue per worker and tasks are queued by the hash of their text, so the tasks
   of a text are processed in submission order. `--learn-queue-capacity` is split between the workers
 - Training holds the model's shared lock (concurrent with other trainers and classifications) so that removing
   counts and memory compaction never race with it

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