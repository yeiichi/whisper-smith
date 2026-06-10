# CHANGELOG

<!-- version list -->

## v0.2.1 (2026-06-10)

### Bug Fixes

- Align Colab GPU torchvision dependency
  ([`2a2fd1b`](https://github.com/yeiichi/whisper-smith/commit/2a2fd1b41b499aec56b43ae0e5781c5a56ffde59))


## v0.2.0 (2026-06-09)

### Bug Fixes

- Replace external image directives with raw HTML for RTD compatibility
  ([`cedcc5c`](https://github.com/yeiichi/whisper-smith/commit/cedcc5c35c332f48b3ff4ee705e48441c65f96da))

- Use code-block:: none for Colab shell magic to avoid lexer warning
  ([`771f338`](https://github.com/yeiichi/whisper-smith/commit/771f338e81d564556aae526c2e0fd54ee7cc92e5))

### Chores

- Add python-semantic-release configuration
  ([`32d3244`](https://github.com/yeiichi/whisper-smith/commit/32d32447368862bb75961a538ee54908c433693c))

- Update uv.lock
  ([`ea8af78`](https://github.com/yeiichi/whisper-smith/commit/ea8af78cba0157b1f21f9a3dcd01548db992fc05))

### Documentation

- Add details on Colab installation and GPU performance in README and docs
  ([`64052de`](https://github.com/yeiichi/whisper-smith/commit/64052de2df611639ba49d1b932c4b08133d6d03c))

- Add Google Colab notebook and Sphinx page for aligned transcript pipeline
  ([`63f98c9`](https://github.com/yeiichi/whisper-smith/commit/63f98c9e2098380f3905e0cfb460a3a53e04f871))

- Improve Colab launcher to avoid Matplotlib GUI and package conflicts
  ([`d04485c`](https://github.com/yeiichi/whisper-smith/commit/d04485cc4ed26dc0c6c5420c814d417f4c02a453))

- Simplify pip install command in Colab examples
  ([`33270ca`](https://github.com/yeiichi/whisper-smith/commit/33270ca583206cc0a958cf2521d1b7c0641a887d))

- Use --target install instead of venv in Colab to avoid environment issues
  ([`6d589ee`](https://github.com/yeiichi/whisper-smith/commit/6d589ee0e7cf9df92100d6d199c90677239b4e39))

- Use venv for Colab installation to avoid package conflicts
  ([`d4ed19f`](https://github.com/yeiichi/whisper-smith/commit/d4ed19fdf9ebc932466bf525c647f9b638ed3294))

### Features

- Support Colab aligned transcript workflow
  ([`d4918ff`](https://github.com/yeiichi/whisper-smith/commit/d4918ffa6ab17b40e92fc14efdc57afdf2681a79))


## v0.1.0 (2026-06-08)

- Initial Release
