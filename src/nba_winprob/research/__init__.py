"""Research and development code — not on any serving path.

Everything in this package was written to answer a question about model choice,
and is kept so those answers stay reproducible. Nothing here is imported by the
API, the streaming processor, or the model that is served; see
``training/logistic.py`` and ``features/basis.py`` for the production path.

The shared evaluation machinery these modules build on — fold construction,
metrics, bootstrap intervals — stays in ``training/advanced.py``, because the
benchmarks the CLI still exposes use it too.
"""
