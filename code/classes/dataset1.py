import pandas as pd
import datasets


class Dataset:

    def __init__(self, path_passage, path_query, path_passages_augmented, path_query_passage):


        self.passages = pd.read_csv(path_passage, sep = '\t')
        self.passages_augmented = dataset = datasets.load_dataset('castorini/msmarco_v1_passage_doc2query-t5_expansions', data_files=path_passages_augmented)

        self.query = pd.read_csv(path_query, sep = '\t')
        self.query_passage= pd.read_csv(path_query_passage, sep = '\t')



    def eval_solution(self, query_passage_predicted):
       


        return "Results"