import re

from .parser import getRelevantSections
from .utils import readJsonFile
from .config import Constants
class Paper:
    """
    The preprocessor class is used to preprocess the dataset.
    It demands the input as the json file of the paper.
    The preprocessor then extracts the various sections. Using
    the `parser` module also extracts the relevant sections.
    """
    def __init__(self, datasetFile : str ) -> None:
        """
        @param datasetFile: The json file of the paper.
        @return: None
        """
        self.datasetFile = datasetFile
        self._file = self.readFile()

    def readFile(self):
        self._file = readJsonFile(self.datasetFile)
        return self._file
    
    def __str__(self) -> str:
        return str(self._file)
    
    def getRelevantContent(self) -> str :
        """
        Returns the relevant content of the paper, 
        in our case this is the sections that can be considered
        as the methodology sections.
        """
        assert(type(self._file) is dict)
        sectionData : dict[str, str] = {} # Maps the section name to the content.
        for section in self._file.get('sections', [] ):
            # Remove numbers from the section name for ease of verification 
            # of the fact that it is part of the paper's methodology section 
            # or not.
            reducedSectionHeading = re.sub(Constants.sectionRenameRegex, '', section['heading'])
            sectionData[str(reducedSectionHeading).lower()] = section['text'].lower()
        
        # Now we check which sections are relevant for us.
        relevantSections = getRelevantSections(list(sectionData.keys())) # Pass the sections we just got.
        releventContent = ' '.join(sectionData.get(relevantSection, Constants.defaultSectionText) for relevantSection in relevantSections)
        return releventContent

    def getPaperData(self) -> dict :
        """
        Returns a dictionary with the following columns :
        'paperID', 'abstract', 'relevantContent', 'title'.
        The paperID is extracted from the path of the paper.
        """
        assert(type(self._file) is dict)
        data = {}
        
        # Extract the paper ID from the file name.
        data['paperID'] = int(self.datasetFile.split('.json')[0].split('/')[-1])
        # The extract is directly accesible through the file.
        data['abstract'] = self._file.get('abstract', '')
        
        # Need to smartly handle the relevant sections.
        data['relevantContent'] = self.getRelevantContent()
        
        # Getting the title from the paper as well.
        data['title'] = self._file.get('title', '')

        return data
