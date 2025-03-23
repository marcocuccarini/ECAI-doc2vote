from pandas import pd
from BM25 import BM25
from dataset import Dataset


class IR:

    def __init__(self, model, dataset):


        self.model=model
        self.dataset=dataset

        #function for the baseline ranking


    def ranking_base(self):

        ranking_dict = ranking_passages(self.dataset.query, self.dataset.passages, top_n)

    return ranking_dict

    def ranking_doc2query(self):


        for id_query, text_query in self.dataset.passages.items():

            





        ranking_dict = ranking_passages(self.dataset.query, self.dataset.passages, top_n)
       


        return "result"



    def ranking_doc2vote(self)



    
       


        return "result"