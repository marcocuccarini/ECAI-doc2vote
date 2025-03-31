import ir_datasets
import pandas as pd
from datasets import load_dataset
from collections import defaultdict


class Dataset:


    def __init__(self, dataset_name="msmarco-passage/dev/small"):

        # Carica il dataset principale da ir_datasets
        dataset = ir_datasets.load(dataset_name)

        # Dizionari per i passaggi e le query

        # id_passage->passage
        self.passages = {doc.doc_id: doc.text for doc in dataset.docs_iter()}
        # id_query->query
        self.query = {query.query_id: query.text for query in dataset.queries_iter()}

        # id_query -> list of relevat pasagges [id_passage1, .....]

        self.query_passage = self.get_query_passage(dataset)

      


    def list_raduce_passage(self):

        list_passage_reduce=[]


        for i in self.query_passage.keys():

            for j in self.query_passage[i]:


                list_passage_reduce.append(j)

             

        self.list_passage_reduce=list_passage_reduce


    def raduce_passage(self):

    #Function that reduce the number of passage only the th relevant one of the query selected

        reduce_passages={}
        reduce_passages_aug={}

        for i in self.query_passage.keys():

            for j in self.query_passage[i]:


                reduce_passages[j]=self.passages[j]

                reduce_passages_aug[j]=self.passages_augmented[j]

        self.reduce_passages =  reduce_passages
        self.reduce_passages_augmented = reduce_passages_aug




    #This function add to the object dataset the augmented passage to the 

    def add_augmented_file(self, path="castorini/msmarco_v1_passage_doc2query-t5_expansions"):

        #dictionary of passage augmeted

        # id_passage -> list of reformulate passage [passage1refomulate1, passage1refomulate2, ...]
        
        #id_passage -> text_passage1+text_passage1refomualte1+text_passage1refomualte2+.......

        dataset1 = load_dataset(path, split="train")
        filtered_dataset = dataset1.filter(lambda item: item["id"] in self.list_passage_reduce)
        self.passages_augmented = {item["id"]: item["predicted_queries"] for item in filtered_dataset}


    def passage_creation_vote(self,aug_num):

        #function that usnify each passage to the realtive genrated query, like in the paper doc2vec

        list_of_value=[]

        for i in self.reduce_passages.keys():

            list_of_value.append([i,self.reduce_passages[i]])

            for j in self.reduce_passages_augmented[i][:aug_num]:

                list_of_value.append([i, j])

        self.list_doc2vote=list_of_value










            




    def join_passage_base_aug(self, aug_sel):

        #function that usnify each passage to the realtive genrated query, like in the paper doc2vec

        dict_temp={}

        for i in self.reduce_passages.keys():

            joined_string = " ".join(map(str, self.reduce_passages_augmented[i][:aug_sel]))

            dict_temp[i]=self.reduce_passages[i]+" "+ joined_string

        self.reduce_passages_base_aug = dict_temp




    def get_query_passage(self, dataset):
        """
        Extracts query-passage relevance mappings.
        """
        query_relevance = defaultdict(list)

        for qrel in dataset.qrels_iter():
            query_relevance[qrel.query_id].append(qrel.doc_id)

        return dict(query_relevance)

   





