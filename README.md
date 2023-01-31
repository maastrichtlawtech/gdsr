<img src="docs/img/icon.png" width=125 height=125 align="right">

[![License: CC BY-SA 4.0](https://img.shields.io/badge/license-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)

# Graph-augmented Dense Statute Retriever

This repository contains the code for reproducing the experimental results presented in the EACL 2023 paper "[Finding the Law: Enhancing Statutory Article Retrieval via Graph Neural Networks](https://arxiv.org/abs/2301.12847)" by [Antoine Louis](https:/antoinelouis.co/work/), [Gijs van Dijck](https://www.maastrichtuniversity.nl/gijs.vandijck) and [Jerry Spanakis](https://dke.maastrichtuniversity.nl/jerry.spanakis/).

<p align="center"><img src="docs/img/cover.png" width="800"></p>

Statutory article retrieval (SAR), the task of retrieving statute law articles relevant to a legal question, is a promising application of legal text processing. In particular, high-quality SAR systems can improve the work efficiency of legal professionals and provide basic legal assistance to citizens in need at no cost. Unlike traditional ad-hoc information retrieval, where each document is considered a complete source of information, SAR deals with texts whose full sense depends on complementary information from the topological organization of statute law. While existing works ignore these domain-specific dependencies, we propose a novel graph-augmented dense statute retriever (G-DSR) model that incorporates the structure of legislation via a graph neural network to improve dense retrieval performance. Experimental results show that our approach outperforms strong retrieval baselines on a real-world expert-annotated SAR dataset.

## Documentation

Detailed documentation on the dataset and how to reproduce the main experimental results can be found [here](docs/README.md).

## Citation

For attribution in academic contexts, please cite this work as:

```latex
@inproceedings{louis2023finding,
  title = {Finding the Law: Enhancing Statutory Article Retrieval via Graph Neural Networks},
  author = {Louis, Antoine and van Dijck, Gijs and Spanakis, Gerasimos},
  booktitle = {Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics},
  month = may,
  year = {2023},
  address = {Dubrovnik, Croatia},
  publisher = {Association for Computational Linguistics},
  url = {},
  pages = {},
}
```

## License

This repository is licensed under the terms of the [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license.
