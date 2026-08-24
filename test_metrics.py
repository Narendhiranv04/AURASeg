import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path('/media/naren/Windows/Users/naren/Documents/AURASeg/benchmark_models')))
from wacv_metrics import compute_metrics

preds = np.array([0, 1, 1])
targets = np.array([0, 1, 0])
m = compute_metrics(preds, targets)
print(m.keys())
