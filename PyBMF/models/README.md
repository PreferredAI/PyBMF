# PyBMF models

Only the fully prepared models are imported in `__init__.py`.

If the model is included in `models/__init__.py`, 
import the models using:
```python
from PyBMF.models import <model_name>
```

If the model is experimental and not included in `models/__init__.py` yet, 
import the models using:
```python
from PyBMF.models.<model_file_name> import <model_name>
```