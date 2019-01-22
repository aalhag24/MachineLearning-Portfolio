# Apriori Associated Rule Learning

# Import the Libraries
import pandas as pd
import os

from apyori import apriori

# Import the data
BinPath = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'bin'))
FilePath = BinPath + '\Market_Basket_Optimisation.csv';
dataset = pd.read_csv(FilePath, header = None)

# Moving the dataset to a list of transaction string
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j] for j in range(0, 20))])

# Training Apriori on the dataset
rules = apriori(transactions, 
                min_support = 0.003, 
                min_confidence = 0.2, 
                min_lift = 3, 
                min_length = 2)

# Visualising the results
results = list(rules)
print(results)