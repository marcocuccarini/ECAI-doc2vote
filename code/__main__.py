
from classes.dataset import Dataset
from classes.evaluator import Evaluator
from classes.model import BM25

import ir_datasets
import json


def extract_json(path):

	 with open(path) as f:

	 	d = json.load(f)
	 	return d


if __name__ == "__main__":


	top_n=10


	dataset_config = extract_json("/Users/marco/Documents/GitHub/ECAI-doc2vote/code/configuration/configDatasetMarco.json")

	dataset=Dataset(dataset_config['dataset_name_test_2'])


	dataset.add_augmented_file()



	dataset.raduce_passage()


	print("Fine")

	#Approach based with model BM25

	model = BM25(list(dataset.reduce_passages.values()))

	#print(ranking_passages)

	score, dict_prediction = model.ranking_score(dataset.query, dataset.reduce_passages, top_n)

	evaluator= Evaluator(dict_prediction,dataset.query_passage)

	rank=evaluator.extract_ranking(top_n)


	print(evaluator.mean_reciprocal_rank(list(rank.values())))

	print("pre-aug")



	print("post-aug")










	































