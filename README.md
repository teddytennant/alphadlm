Sequence-level alpha loss for a factorized masked diffusion denoiser (arXiv 2609.38066).
Install with `pip install -e .`. Tests are `pytest`.
Run the tiny CPU train with `python -m alphadlm.train` (finishes in well under a minute).
Not TinyGSM, not GSM8K 34.6%, not SDAR-1.7B. The denoiser is a tiny MLP, not a transformer, and this package reports no paper benchmark numbers.
