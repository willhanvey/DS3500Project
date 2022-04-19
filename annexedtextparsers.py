import re
import annexedtext as at
from nltk.corpus import stopwords

def remove_arableague_headings(file, removestopwords):
    with open(file, 'r', encoding='UTF-8') as infile:
        # Creating lists to temporarily store data that will be passed to class variables later
        textlist = []
        stats = []
        for line in infile:
            sentence = line.split()
            if sentence:
                # Grabbing words from sentences and culling headings
                if sentence[0] not in ('Topic', 'I.', 'II.', 'III.', 'IV.', 'A.', 'B.', 'C.'):
                    for word in sentence:
                        word = re.sub('[^a-zA-Z]+', '', word).lower()
                        if removestopwords:
                            if word != '' and word not in stopwords.words('english'):
                                textlist.append(word)
                        else:
                            if word != '':
                                textlist.append(word)

    return textlist, stats