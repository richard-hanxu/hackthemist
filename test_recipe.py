from utils import *
from outliers import smirnov_grubbs as grubbs
import numpy as np
# for l in open('labels.txt').readlines():

s = []
w = []
for recipe, weight in get_ingredients('Taco'):
    s.append(z := carbon_footprint(recipe))
    w.append(weight)
print(np.mean(grubbs.test([ss for ss in s], alpha=0.05))*np.mean(grubbs.test([ss for ss in w], alpha=0.05)))
    
