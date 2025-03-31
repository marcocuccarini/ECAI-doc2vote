import ir_datasets
import pandas as pd
from datasets import load_dataset
from collections import defaultdict

class Evaluator():


    def __init__(self, dict_pred, dict_true):



        self.pred = dict_pred

        self.true = dict_true



    def mean_reciprocal_rank(self, ranks):
        """
        Compute Mean Reciprocal Rank (MRR) considering only ranks within top 10 (RR@10)
        :param ranks: List of ranks where the first relevant result appears
        :return: MRR score considering RR@10
        """
        if not ranks:
            return 0.0  # Handle empty input
        
        filtered_ranks = [rank for rank in ranks if rank <= 10]  # Consider only ranks <= 10
        if not filtered_ranks:
            return 0.0  # If no relevant result appears in top 10, return 0
        
        return sum(1.0 / rank for rank in filtered_ranks) / len(filtered_ranks)


    
    def extract_ranking(self, top_n):

        dict_index={}

        for j in self.pred.keys():

            index=top_n

            for i in range(len(self.pred[j])):

                

                if list(self.pred[j])[i] in self.true[j]:

                    if index>i:

                        index=i+1

            dict_index[j]=index

        return dict_index
