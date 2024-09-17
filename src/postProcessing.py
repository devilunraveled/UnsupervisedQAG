from json import loads as LoadsJson
def extractQAPairs(stringOfListOfJsonObjects : str ) -> list[dict[str, str]] :
    """
    This function extracts QnA pairs from the list of 
    json objects given by the model. Returns the QnA 
    pairs extracted. [] if the output cannot be used 
    to extract the relevant QnA pairs.
    """

    # Remove any thing before the first '['
    # and the last ']'.
    filteredString = stringOfListOfJsonObjects.partition('[')[2].partition(']')[0]
    
    try :
        extractedQnA : list[dict[str, str]] = LoadsJson(filteredString)
        return extractedQnA
    except Exception as e:
        print(e)
        print(f"Received String : {stringOfListOfJsonObjects}, converted to {filteredString}")
        return []
