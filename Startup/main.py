import numpy as np
import torch 
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict
#from tqdm import tqdm # TODO: fix verbose training prints to logger or training bar
import networkx as nx
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# Now df_frame will have the mean PCA1 and PCA2 for each graph_id
from sklearn.cluster import KMeans
import seaborn as sns
import sys

# Get variables from command-line arguments
partition = int(sys.argv[1])
a = int(sys.argv[2])


print(f"Variable a: {a}")
# Rest of the code