import os
from json import loads as LoadsJson

from src.utils import writeJsonFile
from config import directories as Paths


def createRawDataset():
    """
    Creates the dataframe for the papers that include the 
    following columns : 
        'paperID', 'abstract', 'relevantSections', 'title'
    """



def separateOutPaper(paperListFile : str, offset : int ) -> int:
    """
    This function takes the list of papers stored as a list of 
    json objects and then extracts all the individual papars from
    the dataset and stores them in the _papers_ directory.
    
    @param paperListFile: The file containing the list of papers.
    @param offset: The offset to be added to the paper ids.
    @return: The number of papers extracted.
    """
    # Store the number of papers.
    _paperCount = 0
    with open(paperListFile, "r", encoding="utf-8") as file:
        for paperID, line in enumerate(file.readlines()):
            paperData = LoadsJson(line.strip().encode("utf-8"))
            writeJsonFile(f"{Paths.papers}/{paperID + offset}.json", paperData)
            _paperCount += 1

    return _paperCount


if __name__ == '__main__' :
    os.makedirs(Paths.papers, exist_ok=True)

    paperListFileDir = Paths.limitGenData
    
    numPapers = 0 # Setting the number of papers yet done to 0.
    for paperListFile in os.listdir(paperListFileDir):
        numExtractedPapers = separateOutPaper(os.path.join(paperListFileDir, paperListFile), numPapers)
        numPapers += numExtractedPapers
    
    print(f"Extracted {numPapers} papers.")
