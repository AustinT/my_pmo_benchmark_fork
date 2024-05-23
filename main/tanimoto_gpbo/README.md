This is basic Tanimoto GP-BO, using a (count) fingerprint Tanimoto GP, a genetic algorithm acquisition function, and a small batch size.

Running it requires:
- the `mol_ga` library, which can be installed with pip <https://github.com/AustinT/mol_ga>
- The GP implementation from my _Tanimoto Random Features_ paper, found here: <https://github.com/AustinT/tanimoto-random-features-neurips23>. It is sufficient to clone the repo and add it to the `PYTHONPATH`.
- `pytorch`, `gpytorch`, and `botorch` (for the library above)
