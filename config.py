import os



class directories:
    currDirectory = os.path.dirname(__file__)
    datasets = os.path.join(currDirectory, "datasets")
    papers = os.path.join(currDirectory, "papers")

    limitGenData = os.path.join(datasets, "limitGenDataset")
