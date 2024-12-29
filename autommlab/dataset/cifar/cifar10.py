import numpy as np

class CIFAR10():

    def __init__(self,
                 fine_label_path='cifar10_fine_label.txt',
                 coarse_label_path='cifar10_coarse_label.txt') -> None:
        self.labels = {'fine':[],'coarse':[]}
        self.label2id = {'fine':{},'coarse':{}}
        self.labels['fine'], self.label2id['fine'] = self.read_label(fine_label_path)
        self.labels['coarse'], self.label2id['coarse'] = self.read_label(coarse_label_path)
        
    def read_label(path):
        labels = np.genfromtxt(path,dtype=str,delimiter='\n')
        label2id = {}
        for index,label in enumerate(labels):
            label2id[label] = index
        return labels,label2id
    
    def label2id(self,label,mode='fine'):
        return self.label2id[mode][label]
    
    def search(self, objetcs, mode='fine'):
        res = []
        for label in self.labels[mode]:
            if label in objetcs:
                res.append()

if __name__=='__main__':
    import nltk
    nltk.download()
    from nltk.corpus import wordnet
    word = "nice"
    synonyms = []

    for syn in wordnet.synsets(word):
        for lm in syn.lemmas():
            synonyms.append(lm.name())
    print (set(synonyms))
