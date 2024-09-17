import os

class Model :
    name = 'meta-llama/Meta-Llama-3.1-8B'

    #### Prompt Configuration ####
    promptPrefix = 'Following is the methodology section content from a academic paper from which you have to generate Question-Answer pairs : '
    promptInfix  = '\n This is an overall summary of the paper : '
    promptSuffix = '\n Generate as many Question-Answer pairs as necessary to capture the information in the methodology section. Output them as a python List of pairs (Question,Answer). Output just the list, nothing else.'


class directories:
    currDirectory = os.path.dirname(__file__)
    datasets = os.path.join(currDirectory, "datasets")
    papers = os.path.join(currDirectory, "papers")

    limitGenData = os.path.join(datasets, "limitGenDataset")
