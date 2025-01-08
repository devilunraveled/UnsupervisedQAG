# %%
# %load_ext autoreload
# %autoreload 2

# %%
from utils import createRawDataset
from config import directories as Paths
from pickle import load as Load
from config import Model, Pipeline
import pandas as pd
from sklearn.model_selection import train_test_split

# %%
dataset = createRawDataset()

# %% [markdown]
"""
## Adding gold answers in the dataset.
"""

# %%
def getGoldQnA( paperID ):
    try :
        with open(f"{Paths.papers}/{paperID}.pkl", 'rb') as f:
            goldQnA = Load(f)
            return goldQnA
    except FileNotFoundError:
        return None

# %%
print("Adding gold answers in the dataset...")
dataset['goldQnA'] = dataset['paperID'].apply(getGoldQnA)

# %%
# Remove papers with no gold QnA
dataset = dataset[dataset['goldQnA'].notna()]
print(f"Final dataset size : {len(dataset)}")

print(dataset.columns)
print(dataset.head())

# %%
# Save Dataset
# dataset.to_csv(f"{Paths.data}/data.csv")

formattedDataset = pd.DataFrame()

# Filter out rows where any column is empty.
dataset = dataset[(dataset['goldQnA'] != '') & (dataset['abstract'] != '') & (dataset['relevantContent'] != '')]

# %%
formattedDataset['modelPrompt'] = dataset.apply(lambda x : f"{Pipeline.promptPrefix} {x.loc['relevantContent']} {Pipeline.promptSuffix}", axis = 1)

# %%
# formattedDataset['modelPrompt'] = dataset.apply(lambda x : f"{x.loc['relevantContent']}", axis = 1)

# %%
formattedDataset['gold_QnA'] = dataset.apply(lambda x : x['goldQnA'], axis = 1)
print(formattedDataset['gold_QnA'][0][0].keys())

# %%
# formattedDataset['reconstrunction'] = dataset.apply(lambda x : x['relevantContent'], axis = 1)

# %%
# Perform the necessary splits.
trainDataset, testDataset = train_test_split(formattedDataset, test_size=0.1, random_state=42)

print(trainDataset.columns)
print(testDataset.columns)

# %%
trainDataset.to_csv(f"{Paths.data}/train.csv")
testDataset.to_csv(f"{Paths.data}/test.csv")
