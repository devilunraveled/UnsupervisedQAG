"""
Key observations during the parsing :

1. The section succeding the methodology section is the 
    result/conclusion section.
2. The first section is always abstract by design, the second 
    always contains introduction as a keyword.
3. The methodology section is mostly after the related work section, 
    often characterizes by related and existing keywords.
4. The section preceding methodology is, if possible, a related work 
    section, otherwise regarding datasets, or simply the introduction.
"""
import os
import pickle

from Config import Constants, Data

def getRelevantSections(paperData           : list,
                        startKeyWords       : list = Constants.sectionStartKeywords, 
                        endKeyWords         : list = Constants.sectionEndKeywords, 
                        beforeStartKeyWords : list = Constants.sectionBeforeStartKeywords,
                        defaultSection      : str = Constants.defaultSection
                        ) -> list :
    """
        @params paperData: The sections for the paper.
        @params startKeywords: List of keywords to search for the start.
        @params endKeywords: List of keywords to search for the end.
        @params beforeStartKeyWords: List of words to search before the target section.
        @return: List of relevant sections.
    """
    startSection = 2
    endSection = 3
    ## Relevant sections = [startSection, endSection]
    
    foundStarting = False
    ### Finding the start section
    for i in range(len(paperData)):
        if paperData[i] in startKeyWords:
            startSection = i
            foundStarting = True
            break
    
    ## Did not find any sections with any keywords
    if not foundStarting :
        if ( startSection == 2 ):
            for i in range(len(paperData)):
                if paperData[i] in beforeStartKeyWords:
                    startSection = i + 1
    
    endSection = len(paperData) - 1
    ### Finding the end section
    for i in range(startSection, len(paperData)):
        if paperData[i] in endKeyWords :
            endSection = i
            break
    
    if (startSection >= endSection):
        if defaultSection in paperData:
            return [ defaultSection ]

    return [ paperData[i] for i in range(startSection, endSection) ]


def runRawCheck() -> None:
    datasetDirectory = Data.limitGenData
    
    paperDataDir = os.listdir(datasetDirectory)[0]

    for paperPath in os.listdir(os.path.join(datasetDirectory, paperDataDir)):
        with open(os.path.join(datasetDirectory, paperDataDir, paperPath), "rb") as file:
            paperData = pickle.load(file)
            paperData = list(paperData['sections'].keys())
        sections = getRelevantSections(paperData=paperData)
        if (sections):
            print(f"Start : {sections[0]}, End : {sections[-1]}")
            print("=====================================")
if __name__ == "__main__":
    runRawCheck()
