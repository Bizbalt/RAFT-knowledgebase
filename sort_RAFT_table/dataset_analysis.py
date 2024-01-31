import sort_RAFT_table as srt
import pandas as pd
import numpy as np
import plotly.express as px

# plotting the kinetic curves of the RAFT polymerization

for polymerisation_kinetic in srt.df.iterrows():
    print(polymerisation_kinetic)

