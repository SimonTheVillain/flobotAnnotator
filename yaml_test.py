import yaml
import numpy as np
import sys


test_mat = np.random.rand(10,10)
print(test_mat)
with open('stack21658676.yaml', 'w') as f:
    yaml.dump(test_mat.tolist(), f)


with open('stack21658676.yaml') as f:
    loaded = yaml.load(f)
loaded = np.array(loaded)
print(loaded)