import re

def cleanStrDF(q):
    text = q["query"]
    text = text.replace('\W', ' ')
    text = text.replace('?', '')
    text = text.replace("á", 'a')
    text = text.replace("é", 'e')
    text = text.replace("ö", 'o')
    text = text.replace("Č", 'C')
    text = text.replace("ć", 'c')
    text = text.replace("ó", 'o')
    text = text.replace("ă", 'a')
    text = text.replace("ä", 'a')
    text = text.replace("ü", 'u')
    text = text.replace("ā", 'a')
    text = text.replace("í", 'i')
    text = text.replace("ÿ", 'y')
    text = re.sub('[^0-9a-zA-Z]+', ' ', text)
    return text

def prompt(query):
    input_query=f'''You are a QA Task expert. Generate an answer based on the given query and searched information
    Query: {query}
    Answer:'''
    return input_query