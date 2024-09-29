from json import loads as LoadsJson
def extractQAPairs(output : str ) -> list[dict[str, str]] :
    """
    This function extracts QnA pairs from the list of 
    json objects given by the model. Returns the QnA 
    pairs extracted. [] if the output cannot be used 
    to extract the relevant QnA pairs.
    """
    try :
        filteredString = output[output.find('['):output.rfind(']')+1] 
        extractedQnA : list[dict[str, str]] = LoadsJson(filteredString)
        return extractedQnA
    except Exception as e:
        print(e)
        print(f"Received String : {output}.")
        return []
