## **Simple Summary**

Understanding how cells develop into different types during growth and disease is crucial for advancing medicine, but current methods cannot pinpoint exactly when and where cells make these critical decisions. We developed a new artificial intelligence tool called single-cell reinforcement learning that treats cell development like a strategic decision-making game. Just as a chess player learns to make optimal moves, our system learns to identify the precise moments when cells decide their future fate—whether to become blood cells, immune cells or other specialized types. We tested this approach on various biological systems, including normal human blood cell development, cancer cells, mouse organ development and cells responding to radiation damage. Our method consistently outperformed fifteen existing state-of-the-art tools and successfully identified early decision points that occur before cells show obvious signs of commitment to specific lineages. Additionally, we discovered previously unknown regulatory factors that control these decisions. This breakthrough provides scientists with a powerful new way to understand how cells make developmental choices, which could lead to better treatments for diseases like cancer and improved strategies for regenerative medicine. By revealing the hidden decision-making logic of cellular development, this work opens new possibilities for controlling and directing cell fate in therapeutic applications.

## **Abstract**

Single-cell RNA sequencing now profiles whole transcriptomes for hundreds of thousands of cells, yet existing trajectory-inference tools rarely pinpoint where and when fate decisions are made. We present single-cell reinforcement learning (scRL), an actor–critic framework that recasts differentiation as a sequential decision process on an interpretable latent manifold derived with Latent Dirichlet Allocation. The critic learns state-value functions that quantify fate intensity for each cell, while the actor traces optimal developmental routes across the manifold. Benchmarks on hematopoiesis, mouse endocrinogenesis, acute myeloid leukemia, and gene-knockout and irradiation datasets show that scRL surpasses fifteen state-of-the-art methods in five independent evaluation dimensions, recovering early decision states that precede overt lineage commitment and revealing regulators such as Dapp1. Beyond fate decisions, the same framework produces competitive measures of lineage-contribution intensity without requiring ground-truth probabilities, providing a unified and extensible approach for decoding developmental logic from single-cell data.

## **Documentation**

[![Documentation Status](https://readthedocs.org/projects/scrl/badge/?version=latest)](https://scrl.readthedocs.io/en/latest/?badge=latest)

[documentation](https://scrl.readthedocs.io/en/latest/)

## **Installation**

[![PyPI](https://img.shields.io/pypi/v/singlecellrl.svg?color=brightgreen&style=flat)](https://pypi.org/project/singlecellrl/)

``` bash
pip install singlecellRL
```

## **Reference**
[Fu, Z., Chen, C., Wang, S., Wang, J., & Chen, S. (2025). scRL: Utilizing Reinforcement Learning to Evaluate Fate Decisions in Single-Cell Data. Biology, 14(6), 679.] (https://doi.org/10.3390/biology14060679)
