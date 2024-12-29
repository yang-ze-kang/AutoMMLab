import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from nltk.corpus import wordnet

class DatasetBase():

    CLASSES = None
    
    PALETTE = None

    def __init__(self,
                dataset_name: str='',
                tag: str='',
                dataset_type: str='',
                train_data_prefix: dict={},
                val_data_prefix: dict={}) -> None:
        self.dataset_name = dataset_name
        self.tag = tag
        self.dataset_type=dataset_type
        self.train_data_prefix=train_data_prefix
        self.val_data_prefix=val_data_prefix


    
    def check_hypernym(self, word, candidate_hypernym):
        if not isinstance(word,list):
            word = word.lower().replace(' ','_')
            syns = wordnet.synsets(word, pos=wordnet.NOUN)
        else:
            syns = word
        if not isinstance(candidate_hypernym,list):
            candidate_hypernym = candidate_hypernym.lower().replace(' ','_')
            candidate_hypernyms = set(wordnet.synsets(candidate_hypernym, pos=wordnet.NOUN))
        else:
            candidate_hypernyms = set(candidate_hypernym)
        for syn in syns:
            hypernym_paths = syn.hypernym_paths()
            for path in hypernym_paths:
                if candidate_hypernyms & set(path):
                    return syn
        return False

    def search(self, objects):
        res = {obj:[] for obj in objects}
        for label in self.CLASSES:
            for obj in objects:
                if obj in ['pedestrian','pedestrians','person','persons','pepople','human','humans','athlete','athletes','dancer','dancers','actor','actors','player','players']:
                    if label=='person':
                        res[obj].append(label)
                        break
                    else:
                        continue
                if obj==label or obj[:-1]==label:
                    res[obj].append(label)
                    break
                if self.check_hypernym(label,obj):
                    if isinstance(label,list):
                        res[obj].append(label[0].lemmas()[0].name())
                    else:
                        res[obj].append(label)
                    break
        not_finds = []
        for obj in res:
            if len(res[obj]) == 0:
                not_finds.append(obj)
        if len(not_finds)>0:
            return {'error':not_finds}
        return res
    
    def check(self,objects):
        res = {obj:[] for obj in objects}
        for label in self.CLASSES:
            for obj in objects:
                if self.check_hypernym(label,obj):
                    res[obj].append(label)
                    break
        not_finds = []
        for obj in res:
            if len(res[obj]) == 0:
                not_finds.append(obj)
        if len(not_finds)>0:
            return not_finds
        return True