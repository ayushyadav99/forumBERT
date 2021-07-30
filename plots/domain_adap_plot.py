"""
script to generate plot for RQ1.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("sport_domain_adap_plot.csv", delimiter='\t')
x = list(range(0,16))
training_exs = 62065
x = [training_exs*epoch for epoch in x]
print(x)

for col in df.columns:
    if (col[0] == 'c'):
        plt.scatter(x, df[col], marker='x', color='blue', alpha=0.1)
plt.scatter(x, df['AVG'], marker='s', color='blue')
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()

print(" xmin: {} ; xmax: {}".format(xmin, xmax))
print(" ymin: {} ; ymax: {}".format(ymin, ymax))


plt.plot(x, df["AVG"], label='sport')
plt.plot(x, df["avg+std"], color='blue', alpha=0.1)
plt.plot(x, df['avg-std'], color = 'blue', alpha=0.1)
plt.grid()
plt.fill_between(x, df['avg+std'], df['avg-std'], color='blue', alpha=0.05)


df = pd.read_csv("culture_domain_adap_plot.csv", delimiter='\t')
x = [0,1,2,3,4,5,7,8,9,10,11,12,14]
training_exs =25415 
x = [training_exs*epoch for epoch in x]
print(x)
for col in df.columns:
    if (col[0:4] == 'seed'):
        plt.scatter(x, df[col], marker='+', color='red', alpha=0.1)
plt.scatter(x, df['avg'], marker='*', s = 60, color='red')
plt.xlabel('No. of Finetuned LM Training Sentences', fontsize=11.5)
plt.ylabel('Absolute Accuracy Change', fontsize=11.5)
plt.plot(x, df["avg"], label='kultur')
plt.plot(x, df["avg+std"], color='red', alpha=0.1)
plt.plot(x, df['avg-std'], color = 'red', alpha=0.1)
plt.fill_between(x, df['avg+std'], df['avg-std'], color='red', alpha=0.05)



plt.legend()
plt.show()

