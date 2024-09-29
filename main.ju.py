# %%
# %load_ext autoreload
# %autoreload 2

# %%
from utils import createRawDataset
from config import directories as Paths
from pickle import load as Load

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

# %%
print(dataset.head())

# %%
# Save Dataset
dataset.to_csv(Paths.data)


