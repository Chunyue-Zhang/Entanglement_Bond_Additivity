# Bond Additivity and Persistent Geometric Imprints of Entanglement in Quantum Thermalization

[![arXiv](https://img.shields.io/badge/arXiv-2601.01327-b31b1b.svg)](https://arxiv.org/abs/2601.01327)

This repository contains the main numerical simulation code for the research paper **"Bond Additivity and Persistent Geometric Imprints of Entanglement in Quantum Thermalization"** by Chun-Yue Zhang, Shi-Xin Zhang, and Zi-Xiang Li.

Copyright (c) 2026 Chun-yue Zhang. The code is released under the [Apache License 2.0](LICENSE).

## Repository Structure

The simulation code is organized according to different dynamics within `src/`.

```
src/
├── Floquet/
│   ├── Floquet_EEs_under_symmetry_inequivalent_bipartitions.py
│   └── Floquet_entropies_to_calculate_mutual informations.py
│
├── MBL/
│   ├── MBL_EEs_under_symmetry_inequivalent_bipartitions.py
│   └── MBL_entropies_to_calculate_mutual informations.py
│
├── mixed_field/
│   ├── mixed_field_EEs_under_symmetry_inequivalent_bipartitions.py
│   └── mixed_field_entropies_to_calculate_mutual informations.py
│
├── NN_thermal/
│   ├── NN_thermal_EEs_under_symmetry_inequivalent_bipartitions.py
│   └── NN_thermal_entropies_to_calculate_mutual informations.py
│
├── NNN_thermal/
│   ├── NNN_thermal_EEs_under_symmetry_inequivalent_bipartitions.py
│   └── NNN_thermal_entropies_to_calculate_mutual informations.py
│
└── RQC/
    ├── RQC_EEs_under_symmetry_inequivalent_bipartitions.py
    └── RQC_entropies_to_calculate_mutual informations.py
```

The data for the figures in the main text are within `data/`. They are stored as Python dictionaries in the corresponding .pkl files. Each dictionary contains several keys corresponding to different data arrays. You can load the data using the pickle module, for example:

```python
import pickle

with open('FIG1b.pkl','rb') as f:
    data=pickle.load(f)

print(data)
```

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@misc{zhang2026bondadditivitypersistentgeometric,
      title={Bond Additivity and Persistent Geometric Imprints of Entanglement in Quantum Thermalization}, 
      author={Chun-Yue Zhang and Shi-Xin Zhang and Zi-Xiang Li},
      year={2026},
      eprint={2601.01327},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2601.01327}, 
}
```
