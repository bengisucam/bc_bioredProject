
from nltk.tokenize import sent_tokenize

if __name__ == '__main__':
    sentences = sent_tokenize("Here is the first sentence. Here is the second.Here is the third one.")
    print(sentences)