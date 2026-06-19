# BayesianServer

BayesianServer is a lock-free, incrementally trainable Multinomial Naive Bayes classifier exposed as a high-throughput
HTTP API. It handles multi-label text classification with real-time learning and no downtime.

You can use it as an internal API for programs that need multi-label classification. For example, a messaging app could
plug it in as a spam filter. Users submit examples of spam and not spam, and the model learns the difference over time.

## Table of Contents

<!-- TOC -->
* [BayesianServer](#bayesianserver)
  * [Table of Contents](#table-of-contents)
  * [Installation](#installation)
  * [Usage](#usage)
  * [Configuration](#configuration)
      * [`--model <path>`](#--model-path)
      * [`--archive <path>`](#--archive-path)
      * [`--smoothing <alpha>`](#--smoothing-alpha)
      * [`--memory-limit <MB>`](#--memory-limit-mb)
      * [`--save-interval <sec>`](#--save-interval-sec)
      * [`--read-only <bool>`](#--read-only-bool)
      * [`--host <addr>` / `--port <n>` / `--backlog <n>`](#--host-addr----port-n----backlog-n)
      * [`--threshold <0..1>`](#--threshold-01)
      * [`--http-worker-threads <n>` / `--service-threads <n>`](#--http-worker-threads-n----service-threads-n)
      * [`--max-request-size <size>`](#--max-request-size-size)
      * [`--learner-threads <n>` / `--learn-queue-capacity <n>`](#--learner-threads-n----learn-queue-capacity-n)
      * [`--min-token-length <n>` / `--max-token-length <n>`](#--min-token-length-n----max-token-length-n)
      * [`--cjk-bigrams <bool>`](#--cjk-bigrams-bool)
      * [`--normalize <bool>`](#--normalize-bool)
      * [`--bm25 <bool>`](#--bm25-bool)
      * [`--bm25-k1 <n>`](#--bm25-k1-n)
      * [`--bm25-b <n>`](#--bm25-b-n)
      * [`--online-lr <bool>`](#--online-lr-bool)
      * [`--lr-rate <n>`](#--lr-rate-n)
      * [`--lr-decay <n>`](#--lr-decay-n)
      * [`--label-chain <bool>`](#--label-chain-bool)
      * [`--complement <bool>`](#--complement-bool)
      * [`--tfidf <bool>`](#--tfidf-bool)
      * [`--prior-weight <n>`](#--prior-weight-n)
      * [`--mml <bool>`](#--mml-bool)
      * [`--mml-confidence-threshold <0..1>`](#--mml-confidence-threshold-01)
      * [`--max-docs <n>`](#--max-docs-n)
      * [`--am-enabled <bool>`](#--am-enabled-bool)
      * [`--am-history-size <n>`](#--am-history-size-n)
      * [`--am-capture-rejected <bool>`](#--am-capture-rejected-bool)
      * [`--am-capture-classification <bool>`](#--am-capture-classification-bool)
      * [`--filters <list>`](#--filters-list)
  * [API Reference](#api-reference)
    * [`PUSH /` (training)](#push--training)
    * [`POST /` (classification)](#post--classification)
    * [`GET /` (diagnostics)](#get--diagnostics)
    * [`GET /health` (liveness)](#get-health-liveness)
    * [`GET /analytics` and `POST /analytics` (analytics)](#get-analytics-and-post-analytics-analytics)
    * [Error responses](#error-responses)
  * [Execution Flow](#execution-flow)
  * [Model Implementation](#model-implementation)
    * [Multinomial Naive Bayes](#multinomial-naive-bayes)
    * [Lock-free concurrency](#lock-free-concurrency)
    * [Two scoring formulas from the same counts](#two-scoring-formulas-from-the-same-counts)
    * [Bayesian label dependency chain](#bayesian-label-dependency-chain)
  * [Tokenizer (UnicodeTokenizer)](#tokenizer-unicodetokenizer)
  * [Stop-Word Filtering](#stop-word-filtering)
  * [Tuning](#tuning)
    * [Social-media chat (Twitter/X, Discord, Slack)](#social-media-chat-twitterx-discord-slack)
    * [Livestream chat (Twitch, YouTube)](#livestream-chat-twitch-youtube)
    * [Email](#email)
    * [Customer support tickets](#customer-support-tickets)
    * [Forum / Reddit posts](#forum--reddit-posts)
    * [General tuning tips](#general-tuning-tips)
    * [Threshold calibration](#threshold-calibration)
    * [Feature pruning](#feature-pruning)
    * [Document length normalization](#document-length-normalization)
    * [BM25 term weighting](#bm25-term-weighting)
    * [Online logistic regression stacking](#online-logistic-regression-stacking)
    * [Memory-managed label eviction](#memory-managed-label-eviction)
  * [Folder format](#folder-format)
  * [Periodic persistence](#periodic-persistence)
  * [Learning Queue](#learning-queue)
  * [Threading Model](#threading-model)
  * [Security Considerations](#security-considerations)
  * [References](#references)
* [License](#license)
<!-- TOC -->


## Installation

```bash
git clone https://github.com/nosial/BayesianServer
cd BayesianServer
mvn package
```

Requires JDK 21+. Compiled to Java 21 bytecode for broad compatibility.


## Usage

```bash
# Run with defaults (listens on 0.0.0.0:8080)
java -jar target/bayesian-server.jar

# Train on a spam document
curl -X PUSH http://localhost:8080/ \
  -H 'Content-Type: application/json' \
  -d '{"text":"buy 1 bitcoin get free prostitutes","labels":["spam"]}'

# Train on a ham document
curl -X PUSH http://localhost:8080/ \
  -H 'Content-Type: application/json' \
  -d '{"text":"meeting tomorrow at 3pm","labels":["ham"]}'

# Classify a new document
curl -X POST http://localhost:8080/ \
  -H 'Content-Type: application/json' \
  -d '{"text":"cheap penis pills for sale, show her your true crypto monster"}'
```

## Configuration

The server can be configured using command-line arguments, every option can also be set via an environment variable.

| Option                               | Environment Variable           | Default          | Type              | Description                                                     |
|--------------------------------------|--------------------------------|------------------|-------------------|-----------------------------------------------------------------|
| `--model <path>`                     | `BS_MODEL`                     | `bayesian-model` | Path              | Model directory for persistence                                 |
| `--archive <path>`                   | `BS_ARCHIVE`                   | none             | Path              | Path to a CSV file for archiving training requests              |
| `--host <addr>`                      | `BS_HOST`                      | `0.0.0.0`        | Address           | Bind address                                                    |
| `--port <n>`                         | `BS_PORT`                      | `8080`           | Integer (1-65535) | Bind port                                                       |
| `--backlog <n>`                      | `BS_BACKLOG`                   | `1024`           | Integer           | TCP accept backlog                                              |
| `--threshold <0..1>`                 | `BS_THRESHOLD`                 | `0.5`            | Double (0-1)      | Global multi-label decision threshold                           |
| `--smoothing <alpha>`                | `BS_SMOOTHING`                 | `1.0`            | Double (>0)       | Additive (Lidstone/Laplace) smoothing constant                  |
| `--normalize <bool>`                 | `BS_NORMALIZE`                 | `false`          | Boolean           | L2-normalize input document vectors before scoring              |
| `--memory-limit <MB>`                | `BS_MEMORY_LIMIT`              | `0`              | Integer           | Max heap (MB) for label token data; `0` = unlimited             |
| `--learner-threads <n>`              | `BS_LEARNER_THREADS`           | `2`              | Integer           | Background learning queue workers                               |
| `--learn-queue-capacity <n>`         | `BS_LEARN_QUEUE_CAPACITY`      | `100000`         | Integer           | Max pending learning tasks                                      |
| `--save-interval <sec>`              | `BS_SAVE_INTERVAL`             | `60`             | Integer           | Periodic model persistence interval; `0` disables               |
| `--min-token-length <n>`             | `BS_MIN_TOKEN_LENGTH`          | `2`              | Integer           | Shortest retained token in Unicode code points; `0` = unlimited |
| `--max-token-length <n>`             | `BS_MAX_TOKEN_LENGTH`          | `0`              | Integer           | Longest retained token in Unicode code points; `0` = unlimited  |
| `--cjk-bigrams <bool>`               | `BS_CJK_BIGRAMS`               | `true`           | Boolean           | Emit character bigrams for CJK text                             |
| `--http-worker-threads <n>`          | `BS_HTTP_WORKER_THREADS`       | `auto`           | Integer           | Netty I/O worker threads                                        |
| `--service-threads <n>`              | `BS_SERVICE_THREADS`           | `#cores`         | Integer           | Handler execution threads                                       |
| `--max-request-size <size>`          | `BS_MAX_REQUEST_SIZE`          | `8MB`            | Size string       | Max HTTP request body                                           |
| `--read-only <bool>`                 | `BS_READ_ONLY`                 | `false`          | Boolean           | Load model in read-only mode; disables learning and persistence |
| `--bm25 <bool>`                      | `BS_BM25`                      | `false`          | Boolean           | Enable BM25 term weighting                                      |
| `--bm25-k1 <n>`                      | `BS_BM25_K1`                   | `1.5`            | Double (>=0)      | BM25 term frequency saturation parameter                        |
| `--bm25-b <n>`                       | `BS_BM25_B`                    | `0.75`           | Double (0-1)      | BM25 document length normalization parameter                    |
| `--online-lr <bool>`                 | `BS_ONLINE_LR`                 | `false`          | Boolean           | Enable online logistic regression stacking                      |
| `--lr-rate <n>`                      | `BS_LR_RATE`                   | `0.01`           | Double (>0)       | Initial SGD learning rate for online LR                         |
| `--lr-decay <n>`                     | `BS_LR_DECAY`                  | `0.001`          | Double (>=0)      | Learning rate decay factor for online LR                        |
| `--label-chain <bool>`               | `BS_LABEL_CHAIN`               | `false`          | Boolean           | Enable Chow-Liu tree label chain post-processing                |
| `--complement <bool>`                | `BS_COMPLEMENT`                | `false`          | Boolean           | Enable Complement Naive Bayes scoring                           |
| `--tfidf <bool>`                     | `BS_TFIDF`                     | `false`          | Boolean           | Enable TF-IDF term weighting during classification              |
| `--prior-weight <n>`                 | `BS_PRIOR_WEIGHT`              | `1.0`            | Double (>=0)      | Prior weight multiplier                                         |
| `--mml <bool>`                       | `BS_MML`                       | `false`          | Boolean           | Enable Multi-Model Language mode                                |
| `--mml-confidence-threshold <0..1>`  | `BS_MML_CONFIDENCE_THRESHOLD`  | `0.35`           | Double (0-1)      | Detection confidence below which MML routes to "und" model      |
| `--max-docs <n>`                     | `BS_MAX_DOCS`                  | `0`              | Long (>=0)        | Max documents the model may learn; 0 = unlimited                |
| `--am-enabled <bool>`                | `BS_AM_ENABLED`                | `true`           | Boolean           | Enable analytical monitoring history                            |
| `--am-history-size <n>`              | `BS_AM_HISTORY_SIZE`           | `10000`          | Integer (>=1)     | Max analytics entries to retain before eviction                 |
| `--am-capture-rejected <bool>`       | `BS_AM_CAPTURE_REJECTED`       | `true`           | Boolean           | Capture rejected learning tasks in analytics                    |
| `--am-capture-classification <bool>` | `BS_AM_CAPTURE_CLASSIFICATION` | `false`          | Boolean           | Capture classification requests in analytics                    |
| `--filters <list>`                   | `BS_FILTERS`                   | none             | Comma-separated   | Pre-tokenization filters; use `all` for every filter            |
| `--log-level <level>`                | `BS_LOG_LEVEL`                 | `INFO`           | String            | Logging level: `TRACE`, `DEBUG`, `INFO`, `WARN`, `ERROR`, `OFF` |
| `-h, --help`                         |                                | --               | Flag              | Show usage and exit                                             |

#### `--model <path>`

Filesystem path where the model is persisted. If the path does not exist it is created as a directory.

#### `--archive <path>`

When set, every training request accepted by the `PUSH /` endpoint is appended as a row to the specified CSV file.
The file is created with a header row (`labels,content`) if it does not already exist. I/O errors while writing to the
archive are logged as warnings but do not affect request processing — the training task proceeds normally even if the
archive write fails.

This is useful for auditing, debugging misclassifications, or replaying training data during migration.

#### `--smoothing <alpha>`

Additive smoothing constant (Laplace/Lidstone smoothing). Added to every token count before computing probabilities.
Must be `> 0`. Default `1.0` is standard Laplace smoothing. Lower values like `0.1` make the model more confident but
risk overfitting. Higher values make the model more conservative.

#### `--memory-limit <MB>`

Maximum heap (in megabytes) for per-label token-count data. When `0` (default), all labels stay in memory. When set to a
positive value, Caffeine uses a weight-based eviction policy (estimating ~200 bytes per token) to evict the
least-frequently-used labels to disk. Evicted labels are persisted via `StructureModelStore` and transparently reloaded
on the next access.

In MML mode (`--mml`) the budget is apportioned equally across all per-language models. With `--memory-limit 256` and 5
active languages, each language model gets roughly 51 MB (256 / 5). When a new language is encountered, the budget is
redistributed automatically. This avoids the Nx multiplier that would happen if each language model got the full limit
independently.

#### `--save-interval <sec>`

How often the model is flushed to disk. `0` disables periodic saves. The scheduler uses dirty-checking
(`model.version()`) to skip no-op saves. A final synchronous save always runs on shutdown.

#### `--read-only <bool>`

When `true`, the server loads an existing model but disables all learning and persistence. The `PUSH /` endpoint is
not registered. Useful for deploying a pre-trained model as a pure inference service.

#### `--host <addr>` / `--port <n>` / `--backlog <n>`

Standard TCP listener parameters. `host` controls the bind interface (`0.0.0.0` for all interfaces, `127.0.0.1` for
loopback only). `port` is the HTTP port. `backlog` is the kernel accept queue depth.

#### `--threshold <0..1>`

The default probability cut-off used to build the multi-label prediction set. Each label with one-vs-rest probability
`>= threshold` is included in `predicted_labels`. This can be overridden per-request via the `threshold` field in
`POST /`.

#### `--http-worker-threads <n>` / `--service-threads <n>`

 - `http-worker-threads`: Netty event-loop threads that handle I/O (accept, read, write). `0` means Netty auto-detects
   (2x CPU cores).
 - `service-threads`: The thread pool that executes request handlers (parsing, classification, learning). Defaults to
   the number of CPU cores.

#### `--max-request-size <size>`

Maximum accepted HTTP request body size. Accepts plain bytes or human-friendly suffixes (`KB`, `MB`, `GB`). Requests
larger than this are rejected with HTTP 413.

#### `--learner-threads <n>` / `--learn-queue-capacity <n>`

The learning queue decouples HTTP request latency from model training. The queue is a bounded `ArrayBlockingQueue`.
When full, new training requests are rejected with HTTP 503. `learner-threads` controls how many background workers
drain the queue. `learn-queue-capacity` controls how many `TrainingTask` objects can wait before back-pressure kicks in.

#### `--min-token-length <n>` / `--max-token-length <n>`

Token length bounds in Unicode code points (not bytes). Set to `0` to disable the bound (no minimum / no maximum).
Tokens shorter than `min-token-length` or longer than `max-token-length` are discarded after tokenization.
These are applied after NFKC normalization and lowercasing.

#### `--cjk-bigrams <bool>`

When `true`, the tokenizer emits adjacent character bigrams for continuous scripts (Han, Hiragana, Katakana). This
captures local context in CJK text where words are not whitespace-delimited. When `false`, only per-character unigrams
are emitted.

#### `--normalize <bool>`

When `true`, each document's term-frequency vector is divided by its L2 norm (`sqrt(sum(freq^2))`) before scoring.
This prevents long documents from dominating the probability mass. Disabled by default because it can hurt accuracy on
highly imbalanced datasets where document length correlates with label.

#### `--bm25 <bool>`

When `true`, replaces the raw TF-IDF term weighting with Okapi BM25. BM25 uses non-linear term frequency saturation
and document length normalization to prevent long documents from dominating the score. The formula is:

```
tf  = (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * (docLength / avgdl)))
idf = log((totalDocs - df + 0.5) / (df + 0.5))
weight = tf * idf
```

Disabled by default. Works best on datasets with highly variable document lengths.

#### `--bm25-k1 <n>`

BM25 term frequency saturation parameter. Controls how quickly the term frequency contribution saturates. Higher values
mean raw frequency matters more (linear-like). Lower values mean the model quickly stops gaining signal from repeated
words. Default `1.5` is a widely used default for text. Must be `>= 0`.

#### `--bm25-b <n>`

BM25 document length normalization parameter. Controls how aggressively the model normalizes for document length.
`0.0` means no length normalization. `1.0` means full normalization. Default `0.75` is a widely used default.
Must be in `[0, 1]`.

#### `--online-lr <bool>`

When `true`, enables a per-label online logistic regression layer that calibrates the Naive Bayes probabilities.
The LR model is trained incrementally via SGD on every incoming document. It uses 3 features: NB log-odds, document
token count, and a bias term. The LR-calibrated probability is returned as `lr_probability` in the JSON response and
is used for multi-label thresholding. Disabled by default. Works best when Naive Bayes is over-confident.

#### `--lr-rate <n>`

Initial SGD learning rate for the online logistic regression. Controls how aggressively the LR weights are updated on
each document. Higher values mean faster adaptation but risk instability. Default `0.01` is conservative. Must be `> 0`.

#### `--lr-decay <n>`

Learning rate decay factor for online logistic regression. The learning rate decays as `rate / (1 + decay * t)` where
`t` is the number of updates seen for that label. Higher values mean faster decay (more conservative over time).
Default `0.001` provides very slow decay. Must be `>= 0`.

#### `--label-chain <bool>`

When `true`, the model builds a maximum-weight spanning tree (Chow-Liu tree) from pairwise mutual information collected
during training and corrects per-label log-odds using already-predicted parent labels. Labels that tend to co-occur
receive a boost. Labels that rarely co-occur are suppressed. Single-label data (`top_label`, `top_probability`) comes
from the unchanged base model. Disabled by default.

#### `--complement <bool>`

When `true`, enables Complement Naive Bayes scoring. Instead of computing `P(token | label)`, the model computes
`P(token | ¬label)`. It scores each label by how incompatible it is with the complement distribution. This reduces the
bias toward frequent labels on heavily imbalanced datasets. Disabled by default.

#### `--tfidf <bool>`

When `true`, term weights during classification are multiplied by the inverse document frequency
(`log(totalDocs / df)`), down-weighting tokens that appear across many labels. This is independent of BM25. BM25
replaces TF-IDF entirely, while `--tfidf` applies the classic TF-IDF weighting before scoring. Disabled by default.

#### `--prior-weight <n>`

Multiplier for the multinomial log-prior `logP(L) = priorWeight * log(D_L / D_total)`. Values > 1.0 amplify the
effect of the prior (making label frequency matter more). Values < 1.0 dampen it. A value of `0.0` makes the prior
uniform (all labels equally likely a priori). Default `1.0` is the standard Bayesian prior. Must be `>= 0`.

#### `--mml <bool>`

When `true`, enables Multi-Model Language (MML) mode. Instead of a single global `NaiveBayesModel`, the server creates
one independent model per ISO 639-1 language code. Incoming training and classification requests are routed to the
appropriate language-model via the built-in `LanguageDetection` service. Unknown or undetectable languages fall back
to the `"und"` (undetermined) model.

This is useful when the same label name has different meanings across languages, or when language-specific stop-word
filtering is desired per model. Each language-model maintains its own vocabulary, label set, document counts, and
scoring parameters. They do not share any state.

When combined with `--memory-limit`, the budget is apportioned equally across all per-language models and redistributed
automatically when a new language is encountered. The `GET /` and `GET /health` endpoints return aggregated statistics
across all languages. Disabled by default.

#### `--mml-confidence-threshold <0..1>`

When MML mode is enabled (`--mml true`), this threshold controls how the server handles low-confidence language
detections. The Lingua library returns a confidence score for every detection. Documents whose confidence falls below
this threshold are routed to the `"und"` (undetermined) model during training, preventing ambiguous data from
polluting language-specific models.

During classification, the server uses a meta-classifier that blends the language-specific model and the `"und"`
model based on the detection confidence:

- **Confidence >= 0.95**: Only the language-specific model is used (fast path).
- **Confidence <= 0.05**: Only the `"und"` model is used.
- **Between 0.05 and 0.95**: Per-label probabilities and posteriors are linearly interpolated between the two models.
  The `scoring_method` field in the JSON response is set to `"mml_ensemble"`.

Default `0.35` is a conservative value that routes clearly ambiguous text to the pooled `"und"` model while keeping
confident detections in their language-specific models.

#### `--max-docs <n>`

Maximum number of documents the model is allowed to learn. When the total document count (including any documents
already loaded from disk) reaches or exceeds this value, the server rejects all new training requests with HTTP 503,
behaving like read-only mode for learning. Classification (`POST /`) and diagnostics (`GET /`) continue to work
normally.

Set to `0` (default) for unlimited learning. The limit is only lifted by restarting the server with a higher value or
with `--max-docs 0`.

In MML mode, the limit applies to the **sum** of all per-language models.

#### `--am-enabled <bool>`

Whether to enable the analytical monitoring subsystem. When `true` (default), the server records a bounded history of
training, rejection, and (optionally) classification events. This history can be queried via the `GET /analytics` and
`POST /analytics` endpoints.

When `false`, the monitoring subsystem is completely disabled: no events are recorded and the `/analytics` endpoint
returns an empty result set. This is useful for reducing memory usage and CPU overhead when the monitoring data is not
needed.

#### `--am-history-size <n>`

Maximum number of analytics entries to retain in memory. When the history exceeds this size, the oldest entries are
automatically evicted. Default is `10000`, which is enough for several hours of heavy traffic.

Each entry stores lightweight metadata (timestamps, language code, label names, token counts, etc.) so even the default
size uses only a few megabytes of heap. Must be `>= 1`.

#### `--am-capture-rejected <bool>`

Whether to record rejected learning tasks in the analytics history. When `true` (default), every rejected task (queue full,
max-docs reached, or server shutting down) is recorded with the rejection reason and the detected language.

This is useful for understanding back-pressure patterns and identifying when the server is under heavy load.

#### `--am-capture-classification <bool>`

Whether to record classification requests in the analytics history. When `true`, every `POST /` classification request is
recorded with the detected language, token count, confidence score, and processing latency.

Default is `false` because classification is typically the hottest path and recording every request can add overhead.
Enable this when you want to analyze latency distributions, language detection patterns, or classification throughput.

#### `--filters <list>`

A comma-separated list of pre-tokenization text filters that remove or replace patterns from raw text before
tokenization. Filters are applied in the order specified. Available filter names (case-insensitive):

| Filter                   | Description                                                 |
|--------------------------|-------------------------------------------------------------|
| `email`                  | Removes e-mail addresses (including +labels and subdomains) |
| `url`                    | Removes HTTP/HTTPS/FTP URLs                                 |
| `www`                    | Removes bare `www.` URLs (no protocol)                      |
| `username`               | Removes social-media handles (`@username`)                  |
| `phone`                  | Removes phone numbers (international and local formats)     |
| `credit_card`            | Removes credit-card numbers (Visa, MC, Amex, Discover)      |
| `ip_address`             | Removes IPv4 and IPv6 addresses                             |
| `mac_address`            | Removes MAC addresses                                       |
| `iban`                   | Removes IBANs                                               |
| `crypto_address`         | Removes Bitcoin, Ethereum, and Litecoin addresses           |
| `uuid`                   | Removes UUIDs                                               |
| `hash`                   | Removes hex hashes (MD5, SHA-1, SHA-256, SHA-512)           |
| `emoji`                  | Removes emoji and pictographic symbols                      |
| `html_tag`               | Removes HTML tags                                           |
| `escape_sequence`        | Removes JavaScript escape sequences (`\n`, `\u0020`)        |
| `code_comment`           | Removes C-style block and line comments                     |
| `markdown_link`          | Removes markdown link/image syntax                          |
| `latex`                  | Removes LaTeX math expressions (`$...$`)                    |
| `hex_color`              | Removes hex colour codes (`#RGB`, `#RRGGBB`)                |
| `base64`                 | Removes base64-encoded strings (16+ chars)                  |
| `quoted_string`          | Removes single and double-quoted strings                    |
| `dollar_quoted_string`   | Removes PostgreSQL-style `$$...$$` strings                  |
| `backtick_code`          | Removes backtick-enclosed code blocks                       |
| `json_literal`           | Removes JSON objects and arrays (shallow)                   |
| `html_entity`            | Removes XML/HTML entities (`&amp;`, `&#123;`)               |
| `file_path`              | Removes Unix/Windows absolute file paths                    |
| `cli_flag`               | Removes command-line flags (`-v`, `--verbose`, `/help`)     |
| `legal_symbol`           | Removes copyright/trademark/registered symbols              |
| `repeated_punctuation`   | Removes repeated punctuation (`!!!`, `???`)                 |
| `tab`                    | Replaces tabs with space                                    |
| `carriage_return`        | Removes carriage-return characters                          |
| `windows_line_ending`    | Normalizes CRLF to LF                                       |
| `control_character`      | Removes ASCII control characters (except newline and tab)   |
| `zero_width`             | Removes BOM and zero-width characters                       |
| `directional_formatting` | Removes directional formatting Unicode characters           |
| `variation_selector`     | Removes variation selectors (VS1-VS16)                      |
| `private_use`            | Removes Unicode private-use area characters                 |
| `combining_mark`         | Removes combining diacritical marks                         |
| `non_bmp`                | Removes characters outside the BMP (U+10000+)               |
| `whitespace`             | Normalizes whitespace sequences to a single space           |

Example: `--filters email,url,username,emoji`

A special value `all` applies every filter in the registry sequentially. This is the simplest way to strip all
known PII and noise patterns:

```
--filters all
```

Note that `all` is self-contained — it already includes every individual filter listed above. Combining it with
additional filter names (e.g. `all,email`) will apply those filters twice, which is harmless but redundant.

Each filter is implemented as a compiled `Pattern` and applied in the order listed. The filtered text is then passed to
the tokenizer. This is useful for reducing noise in text classification, especially on user-generated content, social
media data, or web-scraped text.

## API Reference

All endpoints accept and return JSON. Request bodies use snake_case field names. Responses use snake_case field names.
Null fields are omitted from responses.

### `PUSH /` (training)

Learns one or more documents asynchronously. Training happens in background workers and the endpoint returns immediately.

**Single document with one label:**

```json
{"text": "meeting rescheduled to friday", "label": "work"}
```

**Single document with multiple labels:**

```json
{"text": "urgent bug report", "labels": ["work", "urgent"]}
```

**Batch request:**

```json
{"documents": [
  {"text": "alpha release notes", "labels": ["release", "docs"]},
  {"text": "beta crash fix",      "labels": ["bugfix"]}
]}
```

**Response** (`202` accepted / `503` back-pressure):

```json
{"accepted": true, "submitted": 2, "rejected": 0, "pending": 0, "current_docs": 1250, "max_docs": 0, "rejected_max_docs": 0}
```

| Field               | Type    | Description                                  |
|---------------------|---------|----------------------------------------------|
| `accepted`          | boolean | `true` when every task was enqueued          |
| `submitted`         | int     | Number of tasks accepted                     |
| `rejected`          | int     | Number refused because the queue was full    |
| `pending`           | int     | Total tasks currently waiting in the queue   |
| `current_docs`      | long    | Current total documents learned              |
| `max_docs`          | long    | Configured max document limit; 0 = unlimited |
| `rejected_max_docs` | long    | Documents rejected due to max-docs limit     |

Each document may carry one or more labels. Labels are created on first use and never need to be declared in advance.

---

### `POST /` (classification)

```json
{"text": "meeting agenda items"}
```

Optional overrides:

```json
{"text": "cheap viagra offer", "top_k": 5, "threshold": 0.3}
```

**Response:**

```json
{
  "labels": [
    {"label": "spam",  "posterior": 0.87, "probability": 0.94, "log_score": -12.3, "lr_probability": null},
    {"label": "work",  "posterior": 0.13, "probability": 0.21, "log_score": -15.7, "lr_probability": null}
  ],
  "top_label": "spam",
  "top_probability": 0.87,
  "predicted_labels": ["spam"],
  "threshold": 0.5,
  "total_tokens": 3,
  "known_tokens": 3,
  "unknown_token_count": 0,
  "model_version": 42,
  "scoring_method": "naive_bayes",
  "language_code": "en",
  "confidence": 0.98,
  "processing_time_ms": 2
}
```

| Field                             | Type   | Description                                                       |
|-----------------------------------|--------|-------------------------------------------------------------------|
| `labels`                          | array  | Per-label scores sorted by `posterior` descending                 |
| `labels[n].label`                 | string | The label name                                                    |
| `labels[n].posterior`             | double | Multinomial posterior (sums to 1 across labels)                   |
| `labels[n].probability`           | double | One-vs-rest probability (independent per label)                   |
| `labels[n].log_score`             | double | Log-space score used internally                                   |
| `labels[n].lr_probability`        | double | LR-calibrated probability; `null` when online LR is disabled      |
| `top_label`                       | string | Single most probable label                                        |
| `top_probability`                 | double | Posterior of the top label                                        |
| `predicted_labels`                | array  | Labels whose probability meets the effective threshold            |
| `threshold`                       | double | The decision threshold applied                                    |
| `total_tokens`                    | int    | Token count after tokenization                                    |
| `known_tokens`                    | int    | Subset of tokens present in the vocabulary                        |
| `unknown_token_count`             | int    | Subset of tokens not found in the vocabulary                      |
| `model_version`                   | long   | Model version at classification time                              |
| `scoring_method`                  | string | Active scoring pipeline: `naive_bayes`, `naive_bayes+bm25`, `naive_bayes+online_lr`, `naive_bayes+bm25+online_lr`, or `mml_ensemble` (MML mid-confidence only) |
| `language_code`                   | string | Detected language code                                            |
| `confidence`                      | double | Language detection confidence (0..1)                              |
| `processing_time_ms`              | long   | Time taken to classify in milliseconds                            |

`topK` limits the `labels` array. `<= 0` returns all labels.

---

### `GET /` (diagnostics)

Full model and server state.

**Response:**

```json
{
  "uptime_seconds": 843,
  "model": {
    "total_documents": 12500,
    "label_count": 7,
    "vocabulary_size": 436705,
    "total_token_occurrences": 2140000,
    "total_document_tokens": 312500,
    "smoothing_alpha": 1.0,
    "average_document_length": 25.0,
    "average_tokens_per_label": 305714.29,
    "token_density": 0.204,
    "model_version": 42,
    "bm25_enabled": false,
    "online_lr_enabled": false,
    "lr_initial_learning_rate": 0.0,
    "lr_decay_rate": 0.0,
    "bm25_k1": 0.0,
    "bm25_b": 0.0,
    "labels": [
      {"label": "ham",  "document_count": 8000, "total_tokens": 450000, "distinct_tokens": 210000, "document_fraction": 0.64, "avg_token_frequency": 2.14},
      {"label": "spam", "document_count": 4500, "total_tokens": 280000, "distinct_tokens": 150000, "document_fraction": 0.36, "avg_token_frequency": 1.87}
    ]
  },
  "learning": {
    "pending": 0,
    "capacity": 100000,
    "workers": 2,
    "submitted": 12500,
    "processed": 12500,
    "failed": 0,
    "rejected": 0,
    "rejected_max_docs": 0,
    "max_docs": 0,
    "current_docs": 12500
  },
  "server": {
    "default_threshold": 0.5,
    "smoothing_alpha": 1.0,
    "cjk_bigrams": true,
    "min_token_length": 1,
    "max_token_length": 40,
    "current_memory_bytes": 268435456,
    "available_memory_bytes": 2147483648,
    "model_memory_bytes": 1048576,
    "model_memory_limit_bytes": 268435456,
    "read_only": false,
    "mml": false,
    "mml_confidence_threshold": 0.35
  }
}
```

**`model` object:**

| Field                           | Type    | Description                                              |
|---------------------------------|---------|----------------------------------------------------------|
| `total_documents`               | long    | Documents learned (multi-label counted once)             |
| `label_count`                   | int     | Number of distinct labels                                |
| `vocabulary_size`               | long    | Distinct tokens across the whole model                   |
| `total_token_occurrences`       | long    | Sum of all token occurrences across every label          |
| `total_document_tokens`         | long    | Total tokens across all documents                        |
| `smoothing_alpha`               | double  | Additive smoothing constant in effect                    |
| `average_document_length`       | double  | Mean tokens per document                                 |
| `average_tokens_per_label`      | double  | Mean token occurrences per label                         |
| `token_density`                 | double  | Ratio of distinct tokens to total occurrences            |
| `model_version`                 | long    | Model version at snapshot time                           |
| `bm25_enabled`                  | boolean | Whether BM25 term weighting is enabled                   |
| `online_lr_enabled`             | boolean | Whether online logistic regression is enabled            |
| `lr_initial_learning_rate`      | double  | Initial SGD learning rate for online LR                  |
| `lr_decay_rate`                 | double  | Learning rate decay factor                               |
| `bm25_k1`                       | double  | BM25 term frequency saturation parameter                 |
| `bm25_b`                        | double  | BM25 document length normalization parameter             |
| `labels`                        | array   | Per-label breakdown, sorted by document count descending |
| `labels[n].label`               | string  | The label name                                           |
| `labels[n].document_count`      | long    | Documents that included this label                       |
| `labels[n].total_tokens`        | long    | Total token occurrences attributed to this label         |
| `labels[n].distinct_tokens`     | long    | Distinct tokens attributed to this label                 |
| `labels[n].document_fraction`   | double  | Proportion of documents that include this label          |
| `labels[n].avg_token_frequency` | double  | Mean occurrences per distinct token for this label       |

**`learning` object:**

| Field               | Type    | Description                                         |
|---------------------|---------|-----------------------------------------------------|
| `pending`           | int     | Tasks currently waiting to be processed             |
| `capacity`          | int     | Maximum queue capacity                              |
| `workers`           | int     | Number of background learner threads                |
| `submitted`         | long    | Tasks accepted into the queue since startup         |
| `processed`         | long    | Tasks successfully learned since startup            |
| `failed`            | long    | Tasks that threw exceptions while learning          |
| `rejected`          | long    | Tasks refused because queue was full                |
| `rejected_max_docs` | long    | Tasks refused because max-docs limit was reached    |
| `max_docs`          | long    | Max documents the model may learn; 0 = unlimited    |
| `current_docs`      | long    | Current total documents learned                     |

**`server` object:**

| Field                       | Type    | Description                                      |
|-----------------------------|---------|--------------------------------------------------|
| `default_threshold`         | double  | Default multi-label decision threshold           |
| `smoothing_alpha`           | double  | Additive smoothing constant                      |
| `cjk_bigrams`               | boolean | Whether CJK character bigrams are enabled        |
| `min_token_length`          | int     | Shortest retained token length                   |
| `max_token_length`          | int     | Longest retained token length                    |
| `current_memory_bytes`      | long    | Current JVM heap usage (total - free)            |
| `available_memory_bytes`    | long    | Maximum heap JVM will use                        |
| `model_memory_bytes`        | long    | Estimated memory used by loaded label token maps |
| `model_memory_limit_bytes`  | long    | Configured model memory limit; 0 = unlimited     |
| `read_only`                 | boolean | Whether the server is in read-only mode          |
| `mml`                       | boolean | Whether Multi-Model Language mode is enabled     |
| `mml_confidence_threshold`  | double  | Detection confidence threshold for MML routing   |

---

### `GET /health` (liveness)

```json
{"status": true}
```

| Field    | Type    | Description                            |
|----------|---------|----------------------------------------|
| `status` | boolean | `true` when the server is serving      |

---

### `GET /analytics` and `POST /analytics` (analytics)

Returns a paginated, filterable list of analytical monitoring history entries. When analytical monitoring is disabled (`--am-enabled false`), the endpoint returns an empty result set.

**Query parameters** (GET) or **JSON body** (POST):

```json
{"type": "training", "language": "en", "label": "spam", "from": 1690000000000, "to": 1700000000000, "success": true, "limit": 100, "offset": 0, "sort": "desc"}
```

| Parameter  | Type    | Default | Description                                                    |
|------------|---------|---------|----------------------------------------------------------------|
| `type`     | string  | null    | Filter by entry type: `training`, `rejected`, `classification` |
| `language` | string  | null    | Filter by detected language code                               |
| `label`    | string  | null    | Filter by label (entry must contain this label)                |
| `from`     | long    | null    | Minimum timestamp (epoch millis, inclusive)                    |
| `to`       | long    | null    | Maximum timestamp (epoch millis, inclusive)                    |
| `success`  | boolean | null    | Filter by success status (`true`/`false`)                      |
| `limit`    | int     | 100     | Maximum entries to return (1..1000)                            |
| `offset`   | int     | 0       | Number of entries to skip                                      |
| `sort`     | string  | `desc`  | Sort order by timestamp: `asc` or `desc`                       |

**Response:**

```json
{
  "entries": [
    {
      "timestamp": 1700000000000,
      "type": "training",
      "language_code": "en",
      "labels": ["spam"], 
      "token_count": 12, 
      "confidence": 0.95, 
      "processing_time_ms": 3,
      "model_version": 42, 
      "success": true,
      "rejected_reason": null, 
      "text_length": 80
    }
  ],
  "total": 1,
  "returned": 1,
  "offset": 0,
  "limit": 100
}
```

| Field                           | Type    | Description                                                                            |
|---------------------------------|---------|----------------------------------------------------------------------------------------|
| `entries`                       | array   | Matching analytics entries for this page                                               |
| `entries[n].timestamp`          | long    | Epoch milliseconds when the event occurred                                             |
| `entries[n].type`               | string  | Event type: `training`, `rejected`, or `classification`                                |
| `entries[n].language_code`      | string  | Detected language code                                                                 |
| `entries[n].labels`             | array   | Labels associated with the event (null for classification)                             |
| `entries[n].token_count`        | int     | Number of tokens processed (-1 if unknown)                                             |
| `entries[n].confidence`         | double  | Language detection confidence (-1 if unknown)                                          |
| `entries[n].processing_time_ms` | long    | Time taken to process the event in milliseconds (-1 if unknown)                        |
| `entries[n].model_version`      | long    | Model version after the event (-1 if unknown)                                          |
| `entries[n].success`            | boolean | `true` for successful training, `false` for rejected/failed, null for classification   |
| `entries[n].rejected_reason`    | string  | Reason for rejection: `queue_full`, `max_docs`, `shutting_down` (null if not rejected) |
| `entries[n].text_length`        | int     | Length of the input text in characters (-1 if unknown)                                 |
| `total`                         | int     | Total number of matching entries in the history                                        |
| `returned`                      | int     | Number of entries in this page                                                         |
| `offset`                        | int     | The offset applied to the result set                                                   |
| `limit`                         | int     | The maximum page size requested                                                        |

---

### Error responses

All API endpoints return errors in a uniform JSON envelope:

```json
{"error": "...", "status": 400}
```

| Status | Condition                                          |
|--------|----------------------------------------------------|
| 400    | Malformed request body or invalid parameters       |
| 404    | Unknown path                                       |
| 405    | Wrong HTTP method for the path                     |
| 413    | Request body exceeds `--max-request-size`          |
| 500    | Internal server error (unhandled exception)        |
| 503    | Server busy (service pool saturated or queue full) |


## Execution Flow

BayesianServer processes each incoming request through a layered pipeline:

```
+--------------------------------------------------------------------+
|                        HTTP Layer                                  |
|  Netty -> HttpServerInitializer -> HttpRequestDispatcher           |
|  -> HttpRouter -> ApiHandler (ModelInformation/Learning/Classify)  |
+---------------------------+----------------------------------------+
                            |
                            v
+-------------------------------------------------------------+
|                     Service Layer                           |
|  LearningQueue (bounded queue + background workers)         |
|  NaiveBayesModel (lock-free concurrent model)               |
|  PersistenceScheduler (periodic atomic saves)               |
+---------------------------+---------------------------------+
                            |
                            v
+-------------------------------------------------------------+
|                   Persistence Layer                         |
|  ModelStore (interface)                                     |
|    `-- StructureModelStore  (per-label files, partial load) |
+-------------------------------------------------------------+
```

The request processing flow proceeds as follows:

 1. Netty's `HttpServerCodec` decodes raw bytes into HTTP frames on the I/O thread.
 2. `HttpObjectAggregator` assembles chunked bodies up to `--max-request-size`.
 3. `HttpRequestDispatcher` validates the decoder result (400 on malformed), copies request data, and offloads to the
    service thread pool. If the pool is saturated, the request is rejected with 503 immediately.
 4. `HttpRouter` matches the path and method. It throws 404 for unknown paths and 405 for wrong methods.
 5. The matched `ApiHandler` executes on a service thread:
    - Parses the JSON request body via `Json.parse()`
    - Performs domain logic (train, classify, or return diagnostics)
    - Returns an `ApiResponse` envelope
 6. `HttpRequestDispatcher` serialises the response via `Json.toBytes()` and writes it back through Netty.

Three thread tiers are used:
 - **Boss** (1 thread): accepts TCP connections
 - **I/O workers** (`--http-worker-threads`): socket reads and writes only
 - **Service pool** (`--service-threads`): CPU-bound handler logic

The service pool uses `AbortPolicy`. A full pool and queue cause an immediate 503 response.

Persistence runs on an independent background thread. The `PersistenceScheduler` checks the model's version counter
every `--save-interval` seconds and flushes to disk only when the model has changed. A final synchronous save runs
on graceful shutdown.

Learning is decoupled from HTTP handling via the bounded `LearningQueue`. Incoming `PUSH /` requests enqueue
`TrainingTask` objects and return immediately. Background worker threads drain the queue and call `model.train()`.
If the queue is full, the request is rejected with 503.


## Model Implementation

The model is an incrementally trainable, multi-label Multinomial Naive Bayes classifier built for lock-free concurrency.
For every label, it maintains a token-to-count map using `ConcurrentHashMap` and `AtomicLong` (lock-free atomic
counters). Any number of training threads can update the model while readers classify simultaneously without blocking.

During training, text is tokenized through a Unicode-aware pipeline with language-specific stop-word removal. Term
frequencies are computed, and counts are added to every label associated with the document. The model tracks global
aggregates (total documents, token counts, document frequencies, and label co-occurrences) to support features like
BM25 weighting and label chain inference.

Classification produces two complementary probability views from the same counts. A **multinomial posterior**
(softmax-normalized) for ranking the single most likely label, and a **one-vs-rest probability** (sigmoid) for each
label independently. This enables multi-label thresholding. Optional features include BM25 term frequency saturation,
online logistic regression stacking (per-label SGD calibration using log-odds, document length, and bias features),
and a Chow-Liu tree chain classifier that adjusts probabilities based on label dependencies. The model supports tiered
caching (Caffeine L1 memory + disk L2 eviction) and periodic persistence with dirty-checking, all while remaining
continuously available for reads and writes.

The model additionally provides a memory compaction pass that runs automatically before each persistence cycle.
This removes orphaned entries from global maps (document-frequency entries for tokens that no longer appear in any
label) and ensures all aggregate counters are consistent with per-label data.

### Multinomial Naive Bayes

The classifier treats each document as a bag of tokens and computes, for each label *L*:

```
P(L | document) ∝ P(L) * ∏ P(token_i | L)
```

All probabilities are computed in log-space to prevent floating-point underflow on long documents. The model produces
two complementary scores per label:

 - **Multinomial posterior** (softmax over all labels): sums to 1 across labels. Good for single-label arg-max decisions.
 - **One-vs-rest probability** (sigmoid of log-odds against the complement): independent per label. Good for multi-label
   thresholding because each label's score is unaffected by other labels' token counts.

Additive (Laplace/Lidstone) smoothing with configurable `alpha` prevents zero-probability tokens from wiping out a
label's score. Every unseen token contributes a pseudocount of `alpha` to every label.

### Lock-free concurrency

All token counts live in `ConcurrentHashMap<String, AtomicLong>` structures. Any number of learner threads can
increment counts while any number of reader threads classify, all without locks:

 - **Training** increments per-label `AtomicLong` counters atomically.
 - **Classification** iterates over counters with `get()` snapshots. No atomicity required since partially-applied
   training only affects future classifications.

The only synchronized sections are for vocabulary pruning (which must atomically rebuild global tables) and memory
compaction (which must atomically clean up orphaned entries).

### Two scoring formulas from the same counts

Given a label *L* with document count *D_L*, token counts *c_L(t)* for each token *t*, a global document count
*D_total*, and smoothing alpha *alpha*:

**Log-prior:**

```
logP(L) = log(D_L / D_total)
```

**Log-likelihood (with smoothing):**

```
logP(t | L) = log((c_L(t) + alpha) / (Σ c_L(u) + alpha * V))
```

where *V* is the vocabulary size.

**Multinomial posterior:**

```
P(L | doc) = exp(logP(L) + Σ count(t, doc) * logP(t | L)) / Z
```

where *Z* is the partition function (sum over all labels).

**One-vs-rest probability:**

```
odds = exp(logP(L | doc) - log(1 - P(L | doc)))
P_one_vs_rest(L) = odds / (1 + odds)
```

### Bayesian label dependency chain

When the label chain is enabled, the model builds a maximum-weight spanning tree (Chow-Liu tree) from pairwise mutual
information computed from co-occurrence counts. At inference time, labels are processed in breadth-first order of the
tree, and each label's log-odds are corrected using already-predicted parent labels:

```
logOdds(L_i) = baseLogOdds(L_i)
             + Σ I(L_j is predicted) * log(P(L_j=1 | L_i=1) / P(L_j=1 | L_i=0))
```

- Labels that tend to co-occur get a boost when their partner is already predicted.
- Labels that rarely co-occur get suppressed, tightening the multi-label prediction set.

Co-occurrence statistics are collected during training (`labelOccurrence` map) and used only at classification time.
Single-label data is unaffected because `topLabel` and `topProbability` come from the unchanged base model.

## Tokenizer (UnicodeTokenizer)

A Unicode-aware tokenizer that converts text into a list of tokens for model training and classification:

 - **Normalisation**: NFKC normalisation followed by `Locale.ROOT` lower-casing.
 - **Whitespace/punctuation splitting**: for alphabetic scripts (Latin, Cyrillic, Arabic, Hangul, Thai, etc.).
 - **CJK handling**: scripts that do not use spaces (Han, Hiragana, Katakana) are split into character unigrams. When
   `--cjk-bigrams` is enabled, adjacent CJK characters also form overlapping bigrams.
 - **Length filtering**: tokens shorter than `--min-token-length` or longer than `--max-token-length` (measured in
   Unicode code points, not Java `char` units) are discarded.
 - **Empty/blank input** returns an empty list.

## Stop-Word Filtering

BayesianServer filters stop-words during both training and classification to remove common, low-information tokens
before they reach the model. Stop-words are loaded from the embedded [stopwords-json](https://github.com/6/stopwords-json)
git submodule, which provides curated lists for 50 languages.

**Language detection integration:**
 - Every incoming `PUSH /` (training) and `POST /` (classification) request runs the text through `LanguageDetection`
   before processing.
 - The detected ISO 639-1 language code determines which stop-word set is applied.
 - Tokens present in the language-specific stop-word set are discarded after tokenization and never reach the model's
  probability tables.

**Universal fallback:** When language detection returns `"und"` (undetermined), typically for very short text, pure
punctuation, or mixed-language input, the conservative union of every loaded stop-word set is used. This avoids
accidentally discarding meaningful tokens when the language is uncertain.

**Stop-word sets are immutable** once loaded at startup. Language codes are discovered automatically by scanning the
classpath for `stopwords-json/dist/*.json` files. There is no hard-coded list.

In MML mode each per-language model receives language-specific stop-words during training and classification. The
`"und"` fallback model always uses the universal stop-word set.

## Tuning

Below are practical tuning recommendations for different real-world text sources. Each scenario focuses on the tokenizer,
smoothing, scoring, and behavior parameters that directly affect classification accuracy. With additional information
about how model tuning works.

### Social-media chat (Twitter/X, Discord, Slack)

Short, noisy messages with slang, emojis, hashtags, and frequent typos.

```bash
java -jar bayesian-server.jar \
  --model social-model \
  --smoothing 0.5        \
  --threshold 0.35       \
  --min-token-length 2   \
  --normalize false      \
  --label-chain true     \
  --prior-weight 0.8
```

 - `--smoothing 0.5` Lower smoothing makes the model more confident because the vocabulary is small and repetitive.
 - `--threshold` A lower threshold catches more multi-label signals in fragmented sentences.
 - `--min-token-length 2` drops single-character noise (e.g., "u", "r").
 - `--label-chain true` helps because hashtags and mentions often co-occur (e.g., `#work` + `#urgent`).
 - `--prior-weight 0.8` dampens label frequency bias, which is important when trending topics spike and distort the prior.

### Livestream chat (Twitch, YouTube)

Extremely short messages with heavy spam, ASCII art, memes, and copy-paste.

```bash
java -jar bayesian-server.jar \
  --model stream-model \
  --smoothing 1.0        \
  --threshold 0.5        \
  --min-token-length 3   \
  --normalize false      \
  --complement true      \
  --prior-weight 0.5
```

 - `--min-token-length 3` aggressively strips single-character spam ("K", "a", "pog").
 - `--complement true` reduces bias toward frequent spam labels that dominate the chat.
 - `--prior-weight 0.5` weakens the prior because the label distribution is extremely volatile (chat floods).

### Email

Long, structured documents with formal language, subject lines, signatures, and quoted replies.

```bash
java -jar bayesian-server.jar \
  --model email-model \
  --smoothing 1.0        \
  --threshold 0.45       \
  --min-token-length 2   \
  --normalize true       \
  --bm25 true            \
  --bm25-k1 1.2          \
  --bm25-b 0.6           \
  --label-chain true     \
  --prior-weight 1.0
```

 - `--normalize true` prevents long email bodies (with full signatures and quoted threads) from dominating the probability mass.
 - `--bm25 true` with `--bm25-k1 1.2` and `--bm25-b 0.6` handles highly variable document lengths (short subject vs. long body) better than raw TF.
 - `--label-chain true` captures dependencies like `finance` + `urgent` or `invoice` + `payment` that frequently co-occur.

---

### Customer support tickets

Mixed-length technical text with product names, error codes, and urgency indicators. Labels often reflect both topic and severity.

```bash
java -jar bayesian-server.jar \
  --model support-model \
  --smoothing 0.8        \
  --threshold 0.4        \
  --min-token-length 2   \
  --normalize false      \
  --online-lr true       \
  --lr-rate 0.01         \
  --lr-decay 0.001       \
  --label-chain true     \
  --prior-weight 1.0
```

 - `--online-lr true` calibrates probabilities because support tickets often trigger over-confident Naive Bayes scores on rare technical terms.
 - `--label-chain true` models severity-topic co-occurrence (e.g., `critical` + `database` or `low` + `documentation`).
 - `--threshold 0.4` is permissive because multi-label tickets are common (e.g., `bug` + `billing` + `urgent`).
 - `--smoothing 0.8` is slightly lower than default because the vocabulary is stable (product names repeat).

---

### Forum / Reddit posts

Medium-length informal text with markdown, links, and diverse topics. Highly variable document length.

```bash
java -jar bayesian-server.jar \
  --model forum-model \
  --smoothing 1.0        \
  --threshold 0.5        \
  --min-token-length 2   \
  --normalize true       \
  --bm25 true            \
  --bm25-k1 1.5          \
  --bm25-b 0.75          \
  --tfidf false          \
  --prior-weight 1.0
```

 - `--normalize true` + `--bm25 true` handles the wide length range from one-sentence replies to multi-paragraph essays.
 - `--bm25` replaces `--tfidf` because BM25's non-linear saturation better handles forum posts where authors repeat keywords for emphasis.

### General tuning tips

| Goal                            | Parameter to change                          | Direction                  |
|---------------------------------|----------------------------------------------|----------------------------|
| Reduce false positives          | `--threshold`                                | Increase (e.g., 0.5 → 0.7) |
| Catch more labels               | `--threshold`                                | Decrease (e.g., 0.5 → 0.3) |
| Handle long documents           | `--normalize true` + `--bm25 true`           | Enable                     |
| Handle short documents          | `--min-token-length 2` + `--normalize false` | Enable                     |
| Reduce frequent-label bias      | `--complement true` + `--prior-weight 0.5`   | Enable / lower             |
| Calibrate over-confident scores | `--online-lr true`                           | Enable                     |

### Threshold calibration

A fixed global threshold rarely suits every label. `calibrateThresholds` independently optimizes each label's threshold
on a held-out validation set:

```java
model.calibrateThresholds(List.of(
    new NaiveBayesModel.ValidationSample("text A", List.of("label1", "label2")),
    new NaiveBayesModel.ValidationSample("text B", List.of("label3"))
), "f1");  // or "jaccard", "hamming", "accuracy"
```

The method grid-searches 19 thresholds (`[0.05, 0.10, ..., 0.95]`) per label and picks the value that maximizes the
chosen metric. Calibrated thresholds are stored in an internal `ConcurrentHashMap<String, Double>` and are held in
memory only. They are not persisted.

### Feature pruning

`pruneVocabulary(int maxFeaturesPerLabel)` keeps only the top-N most discriminative tokens per label:

```java
model.pruneVocabulary(10_000);  // keep 10k tokens per label
```

The discriminative score for a token *t* in label *L* is the information gain:

```
IG(t, L) = Σ Σ P(t, L) * log(P(t, L) / (P(t) * P(L)))
```

After pruning, the global token table (`globalTokenCounts`, `globalTotalTokens`) is rebuilt from the remaining
per-label data to remain consistent.

### Document length normalization

Pass `--normalize true` on the command line or `normalizeDocumentLength = true` in the constructor. Each document's
term-frequency vector is divided by its L2 norm before scoring, preventing long documents from dominating the
probability mass. Off by default. May hurt accuracy on highly imbalanced datasets.

### BM25 term weighting

Pass `--bm25 true` to replace the raw TF-IDF weighting with Okapi BM25, which uses non-linear term frequency
saturation and document length normalization:

```
tf  = (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * (docLength / avgdl)))
idf = log((totalDocs - df + 0.5) / (df + 0.5))
weight = tf * idf
```

This prevents long documents from unfairly skewing token importance. Tune with `--bm25-k1` (default 1.5) and
`--bm25-b` (default 0.75). Disabled by default.

### Online logistic regression stacking

Pass `--online-lr true` to enable a per-label online logistic regression layer that calibrates the Naive Bayes
probabilities. For each label, a tiny 3-feature LR model is trained incrementally via SGD on every incoming document:

 - Feature 1: NB log-odds `log(P / (1-P))`
 - Feature 2: Document token count
 - Feature 3: Bias `1.0`

The learning rate decays as `rate / (1 + decay * t)` where `t` is the number of updates seen for that label. Memory
footprint is negligible (3 doubles per label). The LR-calibrated probability is returned as `lr_probability` in the
JSON response and is used for thresholding when enabled. Tune with `--lr-rate` (default 0.01) and `--lr-decay`
(default 0.001). Disabled by default.

### Memory-managed label eviction

When `--memory-limit` is set to a positive value, the server enables a two-tier caching strategy:

 - **L1 (memory)**: a Caffeine `LoadingCache<String, ConcurrentHashMap<String, AtomicLong>>` stores the most
   frequently accessed label token-to-count maps in memory.
 - **L2 (disk)**: the `ModelStore` persists evicted label data. `StructureModelStore` supports per-label save and load.

**Eviction flow:**

 - Caffeine's `maximumWeight` is set from `--memory-limit`. When the total weight of cached entries exceeds this
   threshold, Caffeine evicts the least-frequently-used label.
 - The `evictionListener` callback calls `modelStore.saveLabel(label, snapshot)` to persist the evicted data to disk
   before dropping it from memory.
 - On the next access (classify or train), the `CacheLoader` calls `modelStore.loadLabel(label)` to reload the label's
   token counts back into the cache.
 - Labels that were not evicted remain hot in memory and incur zero I/O.

`StructureModelStore` provides full support. Evicted labels are written to individual `.bin` files under
`<model-path>/labels/` and reloaded on demand.

MML mode each per-language model has its own independent Caffeine cache. The shared `--memory-limit` budget is
apportioned equally across all language models so the total aggregate memory stays close to the configured limit.

**Durability note**: The eviction listener writes data synchronously on the calling thread. A crash during eviction
may lose the most recently evicted label's data. The periodic `PersistenceScheduler` (full model save every
`--save-interval` seconds) bounds this window.

In addition, a memory compaction pass runs automatically before each persistence cycle in both single-model and MML
modes. This pass removes orphaned entries from global maps (e.g., document-frequency entries for tokens that no longer
appear in any label) and ensures all aggregate counters are consistent with per-label data. This is a safe operation
that preserves all learned information and does not alter classification output.

## Folder format

A directory with separate files for each component, enabling partial loading and per-label eviction:

```
<model-path>/
|-- metadata.bin              (totalDocuments as a single long)
|-- labels/
|   |-- index.json            (label name to filename mapping)
|   |-- <encoded-name>.bin    (one per label: documentCount + token/count pairs)
|   `-- ...
|-- df.bin                    (document frequency map)
`-- cooccurrence/
    |-- docs.bin              (per-label document counts for chain inference)
    `-- pairs.bin             (pairwise co-occurrence counts)
```

Label filenames are encoded via `labelToFileName()`: alphanumeric characters pass through, others become `_%04x` escape
sequences. The index file is a simple JSON object with a `labels` array of `{name, file}` entries.

Saves are written to a `.tmp` sibling directory first, then atomically moved into place with
`StandardCopyOption.ATOMIC_MOVE` (with fallback for cross-filesystem moves).

**Interface (`ModelStore`):**

| Method                             | Description                                                                      |
|------------------------------------|----------------------------------------------------------------------------------|
| `exists()`                         | Returns `true` if a readable model exists at the configured path                 |
| `save(NaiveBayesModel)`            | Persists the entire model atomically                                             |
| `load(NaiveBayesModel)`            | Restores the entire model from storage; returns `false` if no data exists        |
| `loadLabel(String)`                | Loads a single label's data into a `LabelSnapshot`                               |
| `saveLabel(String, LabelSnapshot)` | Persists a single label's data (used by the Caffeine eviction listener)          |

## Periodic persistence

The `PersistenceScheduler` runs on a fixed delay (default 60 seconds). It tracks dirty state via
`NaiveBayesModel.version()` (an `AtomicLong` that increments on every training operation). If the version has not
changed since the last save, the tick is a no-op. The server also forces a synchronous save on graceful shutdown after
draining the learning queue.

Before each save, the model runs a memory compaction pass that cleans up orphaned global entries and ensures aggregate
counters are consistent.


## Learning Queue

An asynchronous training pipeline that decouples HTTP request latency from model training:

- **Bounded queue**: backed by an `ArrayBlockingQueue` with configurable capacity (`--learn-queue-capacity`,
  default 100000).
- **Background workers**: `--learner-threads` (default 2) threads pull `TrainingTask` objects from the queue and call
  `model.train()`.
- **Non-blocking submission**: `submit()` uses `offer()`. It returns immediately with `false` if the queue is full,
  causing a 503 response.
- **Throughput counters**: `AtomicLong` fields track `submitted`, `processed`, `failed`, and `rejected` counts.
  Exposed via `status()` as `LearningQueueStatus`.
- **Graceful shutdown**: `close()` stops accepting new tasks, drains remaining items from the queue, and joins worker
  threads with a 30-second timeout.

Each `TrainingTask` carries a single document's text and associated labels. The `LearningHandler` (`PUSH /`) splits
batch requests into individual tasks before submission.


## Threading Model

| Pool                  | Threads                                   | Purpose                                                                     |
|-----------------------|-------------------------------------------|-----------------------------------------------------------------------------|
| Netty boss            | 1                                         | Accept TCP connections                                                      |
| Netty I/O workers     | `--http-worker-threads` (auto = 2x cores) | Socket reads and writes                                                     |
| Service pool          | `--service-threads` (default = #cores)    | Handler execution (JSON parse, model classify/train, response construction) |
| Learning workers      | `--learner-threads` (default 2)           | Background model training                                                   |
| Persistence scheduler | 1                                         | Periodic model save (every `--save-interval` seconds)                       |

The service pool uses a synchronous queue with `AbortPolicy`. If all service threads are busy and the queue is full,
the request is rejected with 503 Service Unavailable.

The persistence scheduler runs a memory compaction pass before each save, which operates under the model's write lock.
This briefly blocks concurrent reads, but the lock is held only for the duration of the compaction (typically
milliseconds).


## Security Considerations

BayesianServer is designed as an internal API service and does not implement authentication, encryption, or CORS
headers. The following security considerations apply:

- **No built-in auth**: The server does not authenticate requests. Deploy behind a reverse proxy (for example nginx or
  Envoy) for access control.
- **No TLS**: HTTP traffic is unencrypted. Use a TLS-terminating proxy for production deployments.
- **No CORS**: No `Access-Control-*` headers are set. Requests from browser-based clients will be blocked by
  same-origin policy. Add CORS headers at the proxy layer if needed.
- **Input validation**: All API inputs are validated. Malformed JSON, missing fields, out-of-range parameters, and
  excessively large request bodies are rejected with appropriate 4xx status codes.
- **Resource limits**: `--max-request-size` bounds memory per request. `--memory-limit` caps the model's heap usage.
  The service pool uses `AbortPolicy` to shed load under saturation.
- **Read-only mode**: `--read-only` disables training and persistence, suitable for deploying pre-trained models as
  pure inference services.

## References

Below are links to documents and research papers used as references to build the project.

**Multinomial Naive Bayes**
- [Naive Bayes and Text Classification I - Introduction and Theory](https://arxiv.org/abs/1410.5329) -- Raschka (2014).
  Comprehensive tutorial covering the multinomial event model and Laplace smoothing.
- [A Comparison of Event Models for Naive Bayes Text Classification](https://www.cs.cmu.edu/~dgovinda/pdf/multinomial-aaaiws98.pdf)
  -- McCallum & Nigam (1998). The canonical paper defining the multinomial, multivariate Bernoulli, and two-event
  models used in the classifier.
- [Tackling the Poor Assumptions of Naive Bayes Text Classifiers](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf)
  -- Rennie et al. (2003). Introduces Complement Naive Bayes, used when `--complement` is enabled.

**Chow-Liu Tree (Label Chain Classifier)**
- [Approximating Discrete Probability Distributions with Dependence Trees](https://ieeexplore.ieee.org/document/1054142)
  -- Chow & Liu (1968). The foundational algorithm for building maximum-weight spanning trees from mutual information,
  used in `chowLiuOrdering()`.

**Term Weighting**
- [The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf)
  -- Robertson et al. (2009). Comprehensive review of BM25 by the original authors.
- [Term-Weighting Approaches in Automatic Text Retrieval](https://doi.org/10.1016/0304-4573(88)90021-0) -- Salton &
  Buckley (1988). The classic reference for TF-IDF weighting, used when `useTfIdf` is enabled.

**Online Learning**
- [Online Learning and Stochastic Approximations](https://leon.bottou.org/publications/pdf/online-1998.pdf) -- Bottou
  (1998). The canonical reference for online SGD, used to train the per-label logistic regression calibration models.

**Datasets**
- [SMS Spam Collection](http://archive.ics.uci.edu/dataset/228/sms+spam+collection) - The SMS Spam Collection is a
  public set of SMS labeled messages that have been collected for mobile phone spam research by Tiago Almeida and
  Jos Hidalgo.
- [stopwords-json](https://github.com/6/stopwords-json) - Stopwords for 50 languages in JSON format.


# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.