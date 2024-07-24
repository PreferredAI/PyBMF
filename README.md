# PyBMF

[![Documentation Status](https://readthedocs.org/projects/pybmf/badge/?version=latest)](https://pybmf.readthedocs.io/en/latest/?badge=latest)


A Python library for Boolean Matrix Factorization.
Work under [Preferred.ai](https://preferred.ai/).

**PyBMF is under active development.** We welcome the authors of BMF papers and those interested in BMF to play around and contribute. Please [contact us](nie.ht@outlook.com) if you have any questions or suggestions.


# Prospectives

Boolean matrix factorization (BMF) is a well-known problem in pattern mining. Throughout the years of prosperous research, it has evolved from greedy heuristics to include a wide range of advanced technologies. We hold the belief that a playground with fairness and adaptiveness is necessary for the development of such algorithms.

PyBMF aims to provide a unified framework with:

1. generators for various types of synthetic data
2. unified ways of importing real-world data
3. data splitting and cross-validation utilities
4. negative sampling utilities for continuous methods
5. the ability to utilize sparse matrices for heuristics
6. evaluation tools for binary and continuous metrics
7. visualization tools for single or multi-matrix data
8. tools for saving and loading models and logs
9. ability to incorporate Boolean matrix simplification and visualization models


# Models

| **Category**        | **Model**              | **Paper**                                                                                                                        | **Original Implementation**                                                                 | **In PyBMF**                          |
|---------------------|------------------------|----------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|---------------------------------------|
| Heuristics          | Asso                   | [PKDD2006](https://cs.uef.fi/~pauli/papers/dbp.pdf) [TKDE2008](http://doi.ieeecomputersociety.org/10.1109/TKDE.2008.53)          | [C](https://cs.uef.fi/~pauli//src/DBP-progs/)                                               | ✅                                     |
| Heuristics          | Hyper/Hyper+           | [SIGKDD2011](https://link.springer.com/article/10.1007/s10618-010-0203-9)                                                        |                                                                                             | ✅                                     |
| Heuristics          | GreConD                | [JCSS2010](https://www.sciencedirect.com/science/article/pii/S0022000009000415)                                                  | [MATLAB](https://github.com/martin-trnecka/matrix-factorization-algorithms)                 | ✅                                     |
| Heuristics          | Panda                  | [ICDM2010](https://epubs.siam.org/doi/abs/10.1137/1.9781611972801.15)                                                            |                                                                                             | ✅                                     |
| Heuristics          | Panda+                 | [TKDE2013](https://ieeexplore.ieee.org/abstract/document/6682889/)                                                               |                                                                                             | ✅                                     |
| Heuristics          | NASSAU                 | [SDM2015](http://dx.doi.org/10.1137/1.9781611974010.37)                                                                          | [link](https://cs.uef.fi/~pauli/nassau/)                                                    |                                       |
| Heuristics          | GreConD+               | [DAM2018](https://www.sciencedirect.com/science/article/pii/S0166218X18303755)                                                   | [MATLAB](https://github.com/martin-trnecka/matrix-factorization-algorithms)                 | ✅                                     |
| Heuristics          | MEBF                   | [AAAI2020](https://ojs.aaai.org/index.php/AAAI/article/view/6072/5928)                                                           | [R](https://github.com/clwan/MEBF)                                                          | ✅                                     |
| Continuous          | NMFSklearn             |                                                                                                                                  |                                                                                             | 🛞 Wrapper of sklearn.nmf             |
| Continuous          | WNMF                   |                                                                                                                                  |                                                                                             | ✅ Multiplicative update               |
| Continuous          | BinaryMF-Penalty       | [ICDM2007](https://ieeexplore.ieee.org/abstract/document/4470263/)                                                               | [MATLAB](https://github.com/ZhongYuanZhang/BMF)                                             | ✅ Multiplicative update               |
| Continuous          | BinaryMF-Thresholding  | [ICDM2007](https://ieeexplore.ieee.org/abstract/document/4470263/)                                                               | [MATLAB](https://github.com/ZhongYuanZhang/BMF)                                             | ✅ Line search                         |
| Continuous          | FastStep               | [PAKDD2016](https://link.springer.com/chapter/10.1007/978-3-319-31753-3_37)                                                      | [C++](http://cs.cmu.edu/~maraujo/faststep/)                                                 | ✅ Line search                         |
| Continuous          | PRIMP                  | [DMKD2017](https://dl.acm.org/doi/abs/10.1007/s10618-017-0508-z)                                                                 | [CUDA C++](https://sfb876.tu-dortmund.de/primp/index.html)                                  | ✅ PALM                                |
| Continuous          | PNL-PF                 | [SP2021](https://www.sciencedirect.com/science/article/pii/S0165168420303534)                                                    |                                                                                             | ✅ Multiplicative update               |
| Continuous          | ELBMF                  | [NIPS2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/1e8730e2ccd6cefcf70a98dd90d9af6a-Abstract-Conference.html) | [Julia](https://eda.rg.cispa.io/prj/elbmf/) [Python](https://github.com/sdall/elbmf-python) | ✅ PALM                                |
| Probablistic        | MessagePassing         | [ICML2016](http://proceedings.mlr.press/v48/ravanbakhsha16.html)                                                                 | [Python](https://github.com/mravanba/BooleanFactorization)                                  | 🛞 Wrapper of original implementation |
| Probablistic        | OrMachine              | [ICLM2017](https://proceedings.mlr.press/v70/rukat17a.html)                                                                      | [Cython](https://github.com/TammoR/OrMachine/)                                              | 🛞 Wrapper of original implementation |
| Linear Optimization | ColumnGeneration       | [AAAI2021](https://ojs.aaai.org/index.php/AAAI/article/view/16500/16307)                                                         | [Python](https://github.com/kovacsrekaagnes/rank_k_Binary_Matrix_Factorisation)             | 🛞 Wrapper of original implementation |
| Satisfiability      | UndercoverBMF          | [AAAI2021](https://ojs.aaai.org/index.php/AAAI/article/view/16500/16307)                                                         | [C++](https://github.com/FlorentAvellaneda/UndercoverBMF)                                   | 🛞 Wrapper of original implementation |
| Simplification      | IterEss                | [IS2019](https://www.sciencedirect.com/science/article/pii/S0020025519301902)                                                    |                                                                                             |                                       |
| Simplification      | DelegationBMF          | [AAAI2024](https://ojs.aaai.org/index.php/AAAI/article/view/30049)                                                               | [C++](https://github.com/FlorentAvellaneda/Delegation_BMF)                                  |                                       |
| Visualization       | OrderedBMF             | [SIAM2019](https://doi.org/10.1137/1.9781611975673.82)                                                                           | [C++](https://cs.uef.fi/~pauli/bmf/ordered_bmf/)                                            |                                       |
| Visualization       | BiclusterVisualization | [PKDD2023](https://cs.uef.fi/~pauli/papers/marette23visualizing.pdf)                                                             | [Python](https://github.com/tmarette/biclusterVisualization)                                |                                       |

# Compatibility

Currently built and tested on Python 3.9.18.

# TODO

- [ ] Add mask parameter W to PRIMP and ELBMF
- [ ] Fix DataFrame display utils in dataframe_utils.py
- [ ] Include BMF visualization models
- [ ] Diagnosis of thresholding models