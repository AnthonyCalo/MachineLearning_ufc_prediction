import pandas as pd
import numpy as np
from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_gamma_deviance
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

ufc = pd.read_csv("./ufcData.csv")

ufc['date'] = pd.to_datetime(ufc["date"], format="%Y-%m-%d")
# ufc = ufc.loc[ufc["date"]> datetime.strptime("01/01/2010", "%m/%d/%Y")]
ufc = ufc[["R_fighter", "B_fighter"]]

g=nx.from_pandas_edgelist(ufc, source='R_fighter', target='B_fighter')
degree = sorted(g.degree, key= lambda x: x[1], reverse=True)[:10]
print(degree)
eg_cen = nx.eigenvector_centrality(g)
counter=0
print(type(eg_cen))
# for hero in sorted(eg_cen, key = eg_cen.get , reverse=True):
#     counter+=1
#     print(hero, "EC: {}".format(eg_cen[hero]))    
#     if(counter==10):
#         break

eg_cenTen = sorted(eg_cen, key=eg_cen.get, reverse=True)
fighters = list(eg_cenTen)[:10]
values = [eg_cen[fighters[i]] for i in range(10)]
plt.bar(range(10), values, tick_label=fighters)
plt.title("Fighters Eigen Centrality")
plt.show()
deg_cen = nx.degree_centrality(g)


counter=0

for fighter in sorted(deg_cen, key = deg_cen.get , reverse=True):
    counter+=1
    print(fighter, "DC: {}".format(deg_cen[fighter]))    
    if(counter==11):
        break





