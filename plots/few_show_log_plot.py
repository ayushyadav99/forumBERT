"""
script to generate plot for RQ2
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import pandas as pd


x = [1
, 1.204119983
, 1.342422681
, 1.505149978
, 1.643452676
, 1.806179974
, 1.954242509
, 2.10720997
, 2.255272505
, 2.408239965
, 2.558708571
, 2.709269961
, 2.859738566
, 3.010299957
, 3.160768562
, 3.311329952
, 3.461798558
, 3.612359948
, 3.762828553
, 3.913389944
, 4.063858549
, 4.214419939
, 4.364926034
, 4.45700334
, 4.549346636
, 4.653463367]

x = [
10,
16,
22,
32,
44,
64,
90,
128,
180,
256,
362,
512,
724,
1024,
1448,
2048,
2896,
4096,
5792,
8192,
11584,
16384,
23188,
27987,
34773,
44371,
    ]

#vanilla_df = pd.read_csv('few_shot_vanilla.csv', names=['v1', 'v2', 'v3'], delimiter='\t') 
domain_df = pd.read_csv('few_shot_domain.csv', names=['d1', 'd2', 'd3', 'm', 'm+std', 'm-std'],delimiter='\t')
vanilla_df = pd.read_csv('few_shot_vanilla.csv', names=['v1', 'v2', 'v3', 'm','m+std', 'm-std'], delimiter='\t') 

vanilla_cols = ['v1', 'v2', 'v3']
domain_cols = ['d1', 'd2', 'd3']
#plt.plot(x, mean, color='red')
#plt.fill_between(x, mean_stddev_, mean_stddev, color='orange', alpha=0.4)

#plt.plot(x, domainmean, color='blue')
#plt.fill_between(x, domainmean_stddev_, domainmean_stddev, color='#add8e6', alpha=0.4)
fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
for col in vanilla_cols:
    ax.scatter(x, vanilla_df[col], marker='x', color='red', alpha=0.3)

ax.plot(x, vanilla_df.m, color='red', label='bert-base-german-cased')
ax.scatter(x, vanilla_df.m, marker='*', color='red', s=60)
ax.fill_between(x, vanilla_df['m-std'], vanilla_df['m+std'], color='orange', alpha=0.4)

ax.plot(x, domain_df.m, color='blue', label='topic adapted bert')
ax.scatter(x, domain_df.m, marker='s', color='blue', s=20)
ax.fill_between(x, domain_df['m-std'], domain_df['m+std'], color='#add8e6', alpha=0.4)
for col in domain_cols:
    ax.scatter(x, domain_df[col], marker='+', color='blue', alpha=0.3)

ax.set_xlabel(r'number of downstream training examples', fontsize=12.5)
ax.set_ylabel('f1-score (minority class)', fontsize=12.5)
ax.set_title(r'f1-score vs $log_{10}$(number of downstream training examples)')
ax.set_xscale('log')
ax.set_xticks([10, 30, 100, 315, 1000, 3150, 10000, 31500,  ])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.grid()
#plt.xlabel(r'log_{10}(number of classification examples)')
#plt.ylabel(r'\textbf{f1 score}')
plt.legend()
plt.show()
