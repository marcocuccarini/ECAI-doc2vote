
from classes.dataset import Dataset
from classes.BM25 import BM25

import ir_datasets
import json


def extract_json(path):

	 with open(path) as f:

	 	d = json.load(f)
	 	return d



if __name__ == "__main__":


	dataset_config = extract_json("/Users/marco/Documents/GitHub/ECAI2025doc2vote/code/configuration/configDatasetMarco.json")

	dataset=Dataset(dataset_config['dataset_name_test_2'])

	print("Fine")


	model=BM25(list(dataset.passages.values()))

	res=model.ranking_passages(dataset.query, dataset.passages, 10, dataset.query_passage)

	print(dataset.evaluate_ir_system(res))



	print("Fine")
























