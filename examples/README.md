# PyBMF Examples

This directory contains example notebooks that can be used to understand how PyBMF works.

To use the data from `PyBMF.datasets`, you are encouraged to place a `settings.ini` as shown in this directory.
This configures the location of bulky input and output files.

# Debugging in Jupyter Notebook without installation

In Jupyter Notebook, you can include the relative path of the project files in development.

Whether you've installed PyBMF or not, you can use the following command to make the project path prioritized in the search list.

```python
%load_ext autoreload
%autoreload 2

import sys
sys.path.insert(0, '../')
```

Then the models can be imported as introduced in [README](../PyBMF/models/README.md).

In `.py` Python scripts, one shall use the absolute path of the project instead.
