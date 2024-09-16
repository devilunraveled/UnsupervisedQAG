import logging
from json import load as LoadJson, dump as DumpJson

def readJsonFile(filePath : str):
    try :
        with open(filePath, "r", encoding="utf-8") as file:
            fileData = LoadJson(file)
        logging.info(f"File read : {filePath}")
    except FileNotFoundError:
        fileData = None
        logging.error(f"File not found : {filePath}")
    return fileData

def writeJsonFile(filePath : str, data : dict):
    try :
        with open(filePath, "w", encoding="utf-8") as file:
            DumpJson(data, file)
        logging.info(f"File written : {filePath}")
    except FileNotFoundError:
        logging.error(f"File not found : {filePath}")
