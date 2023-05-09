import os
import json

class MMpediaDataset():
    '''
    The MMpedia main class 
    '''

    def __init__(self, root_dir) -> None:
        self.triplets = list()
        self.entity2image = dict()
        self.entitylist = list()
        self.relationlist = list()

        print("Loading MMpedia triplets...")
        with open(os.path.join(root_dir, "MMpedia_triplets.json"), "r", encoding="utf-8") as f:
            self.triplets = json.load(f)

        print("Loading MMpedia image indexs...")
        with open(os.path.join(root_dir, "entity2image.json"), "r", encoding="utf-8") as f:
            self.entity2image = json.load(f)
        
        entities = set()
        relations = set()
        for triplet in self.triplets:
            relations.add(triplet[1])
        self.entitylist = list(self.entity2image)
        self.relationlist = list(relations)


    def load_mapping(self):
        '''
        Load MMpedia image data
        '''
        return self.entity2image


    def load_entities(self):
        '''
        Load MMpedia entities
        '''

        return self.entitylist

    def load_relations(self):
        '''
        Load MMpedia relations
        '''
        
        return self.relationlist
    
    def load_triplets(self):
        '''
        Load MMpedia triplets
        '''
        
        return self.triplets

    def get_entity_img(self, entity = None, entity2image = None) -> list:
        '''
        Get the images that corresponds to the input entity
        '''

        if entity is None and entity2image is None:
            print("Please specify the entity or provide mapping")
            return
        
        return entity2image[entity]
