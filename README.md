# CalciumKit

**CalciumKit** is a Python toolkit developed for the Ramirez Lab to analyze calcium imaging data collected across various modalities, including:

- Fiber Photometry (FP)
- 1-Photon Calcium Imaging (1P)
- 2-Photon Calcium Imaging (2P)

This package is intended to serve as a unified, flexible, and reusable framework for calcium signal preprocessing, analysis, and visualization—tailored to the experimental workflows of the lab. If you are not in the Ramirez Lab, no worries, I will accept pull-requests as long as they follow idiomatic conventions established below. If you have any questions, shoot me an email at rsenne at bu dot edu.

---

## Core Principles

- **Modality-agnostic design**: Functions are designed to work across FP, 1P, and 2P with minimal assumptions.
- **Reproducibility**: Clear data workflows that support repeatable analysis across experiments and users.
- **Extensibility**: Modular structure encourages contributions and lab-specific customizations.
- **Usability**: Designed to be easy to use for lab members with varying levels of coding experience.
- **IO Independence**: IO-independence: Core functions operate on standard Python/numpy data structures (e.g., arrays, DataFrames) and are intentionally decoupled from specific file formats or loading pipelines—data loading is left up to the user. This does not mean that IO functionality can't be included, but rather, that any core functiopns should not rely on the users specific idiosyncratic ways of data loading. 

## Installation
Right now CalciumKit is not registered. To install to a local env do the following:

```bash
pip install git+https://github.com/rsenne/calciumkit.git
```

You may need to clone and install in editable mode if actively developing:

```bash
git clone https://github.com/rsenne/calciumkit.git
cd calciumkit
pip install -e .
```

## Contributions

Contributions from lab members (and others!) are welcome! Please:

- Follow the existing code style (see below)
- Include tests for new features
- Keep functions modular and IO-agnostic where possible
- Open a pull request with a clear description
- Follow PEP-8 conventions where possible

## Code Style

Before submitting a pull request, please run black on your code:

```bash
black calciumkit/ tests/
```

You can install black with:

```bash
pip install black
```

Code that doesn't conform to black style may be rejected by CI or requested to be revised.

