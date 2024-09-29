import os

class Model :
    name = 'llama3.1:8b'

    #### Prompt Configuration ####
    promptPrefix = 'Following is the methodology section content from a academic paper from which you have to generate Question-Answer pairs : '
    promptInfix  = '\n This is an overall summary of the paper : '
    promptSuffix = """Given the above paragraph from a research paper, generate question answer pairs, that capture the information content in the paragraph, output just a json object of Questions and Answers. The structure should be as shown: 
    [
        {
            "question" : "Question 1",
            "answer" : "Answer 1"
        },
        {
            "question" : "Question 2",
            "answer" : "Answer 2"
        },
        {
            "question" : "Question 3",
            "answer" : "Answer 3"
        }
    ]
    You can incorporate as many questions as you deem necessary, but they should be distinct, you are to return nothing other than this json object. If there is any reason due to which you cannot generate the questions, return an empty list. Remember, your output must always be a list of json objects, no verbosity.
        """


class directories:
    currDirectory = os.path.dirname(__file__)
    data = os.path.join(currDirectory, "data")
    datasets = os.path.join(currDirectory, "datasets")
    papers = os.path.join(currDirectory, "papers")
    limitGenData = os.path.join(datasets, "limitGenDataset")

class EncoderConfig :
    Name : str = "bert-base-uncased"

class ReconstructionDecoderConfig :
    Name : str = "gpt2"

class QAGenerationDecoderConfig :
    Name : str = "gpt2"
