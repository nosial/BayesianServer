# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2026-08-16

This update introduces security fixes

### Fixed
 - Made HTTP method routing case-sensitive and added a configurable request read timeout (`--request-read-timeout-ms`) to close stalled connections.
 - Prevented label filename collisions and oversized filename failures with injective escaping and bounded hashed filenames.
 - Capped persistable token length and labels per document; label-pair tracking now runs only when label-chain mode is enabled.


## [1.0.0] - 2026-06-19

Initial release of BayesianServer