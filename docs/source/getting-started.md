Getting Started
===============

EEGProc-XAI-CFO is a fully vectorized library designed for preprocessing and extracting features from EEG(Electroencephalogram) data. This library is optimized for performance and ease of use, making it suitable for researchers and developers working in the field of neuroscience, biomedical engineering, and machine learning.

Installation
------------

Install from the extracted source archive:

```bash
python -m pip install -e .
```

Dependencies
------------

EEGProc-XAI-CFO relies on:

- **NumPy**, **Pandas**, **SciPy** – numerical processing
- **PyWavelets** – wavelet features
- **PyEMD** – empirical mode decomposition
- **Matplotlib** – plotting utilities

Quick Start
-----------

1. **Import and load your EEG data:**

```python
import pandas as pd
from eegproc_xai_cfo import bandpass_filter, FREQUENCY_BANDS

df = pd.read_csv("my_eeg_data.csv")
fs = 128  # Hz
```

2. **Filter and extract features:**

```python
clean = bandpass_filter(df, fs, bands=FREQUENCY_BANDS)
from eegproc_xai_cfo import shannons_entropy, hjorth_params

entropy_df = shannons_entropy(clean, fs)
hjorth_df  = hjorth_params(clean, fs)
```

3. **Visualize results:**

```python
from eegproc_xai_cfo.plotting import plot_per_channel
plot_per_channel(entropy_df, title="Shannon Entropy per Channel")
```
