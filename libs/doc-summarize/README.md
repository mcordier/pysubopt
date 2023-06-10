# doc-summarize

doc-summarize is a Python package that focuses on the task of text summarization. The goal of text summarization is to generate a concise summary that captures the most important information from one or more documents. This package is built upon ubmodular optimization, and is designed to provide summarization.

## Install

To install the doc-summarize package, please follow the setup instructions provided in the mono-repo instructions. These instructions will guide you through the installation process for the entire PySubOpt monorepo, including the DocumentSummarization package.

## Quickstart

To quickly get started with the doc-summarize package, you can use the following command:

```bash
opt-summarize covdiv-greedy --path ./data/facebook_combined.txt
```

This command performs summarization using the `covdiv-greedy` algorithm on the specified input file (`./data/facebook_combined.txt`). You can modify the algorithm or input file based on your requirements.

## Modeling

### Problem Formulation

In the doc-summarize package, the text summarization problem is formulated as follows:

- The ground set $V$ represents all the sentences in a document.
- For extractive document summarization, the objective is to select a small subset $S$ that accurately represents the document (ground set $V$).
- The summary is required to be length-limited, where:
    - $c(S)$ represents the cost for sentences $S$ (e.g., the number of words in all sentences of $S$).
    - $b$ represents the budget, which is the largest length allowed.
- A set function $f: 2^V \rightarrow \mathbb{R}$ is used to measure the quality of the summary $S$.

Thus, the problem can be formulated as follows:

$$
S^* \in \arg\max_{S \subseteq V} f(S) \quad \text{subject to} \quad c(S) \leq B
$$

This problem is known to be NP-hard, and the doc-summarize package provides methods to compute near-optimal strategies.

### Submodular Optimization

In the doc-summarize package, submodular optimization plays a crucial role in the text summarization process. A function $f: 2^V \rightarrow \mathbb{R}$ is considered submodular if, for any subsets $A$ and $B$ of $V$, the following inequality holds:

$$
f(A) + f(B) \geq f(A \cap B) + f(A \cup B)
$$

Submodular functions provide a mathematical foundation for capturing properties such as relevance and redundancy in the summary generation process.

The doc-summarize package leverages submodular optimization techniques to maximize the sum of various submodular functions that represent different constraints, such as relevance and redundancy. A simple greedy algorithm is used to produce an approximately optimal summary based on this framework. The package also introduces a comparison with a more classical approach (MMR) using an accelerated modified greedy algorithm.
