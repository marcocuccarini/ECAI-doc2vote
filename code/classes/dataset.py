import ir_datasets
import pandas as pd
from datasets import load_dataset
from collections import defaultdict

class Dataset:
    def __init__(self, dataset_name="msmarco-passage/dev/small"):
        # Carica il dataset principale da ir_datasets
        dataset = ir_datasets.load(dataset_name)

        # Dizionari per i passaggi e le query
        self.passages = {doc.doc_id: doc.text for doc in dataset.docs_iter()}
        self.query = {query.query_id: query.text for query in dataset.queries_iter()}

        # Carica il file query-passage
        self.query_passage = self.get_query_passage(dataset)

    @staticmethod
    def mean_reciprocal_rank(ranks):
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

    def evaluate_ir_system(self, query_passage_predicted, k=10):
        """
        Evaluates an IR system based on query-passage predictions.
        :param query_passage_predicted: Dictionary of predicted passages for each query
        :param k: Top-k ranking consideration
        """

        list_rank = []
        dict_rank = {}

        for query_id, relevant_docs in self.query_passage.items():
            list_index = []

            if query_id in query_passage_predicted:
                predicted_docs = query_passage_predicted[query_id]

                for relevant_doc in relevant_docs:
                    if relevant_doc in predicted_docs:
                        list_index.append(predicted_docs.index(relevant_doc))
                        
            # If no relevant document is found, assign a rank of 11 (out of top 10)
            dict_rank[query_id] = min(list_index) if list_index else k+1

        list_rank = list(dict_rank.values())


        print(f"Top 10 ranks: {list_rank}")
        mrr_score = self.mean_reciprocal_rank(list_rank)
        print(f"MRR: {mrr_score:.4f}")

    def get_query_passage(self, dataset):
        """
        Extracts query-passage relevance mappings.
        """
        query_relevance = defaultdict(list)

        for qrel in dataset.qrels_iter():
            query_relevance[qrel.query_id].append(qrel.doc_id)

        return dict(query_relevance)

    def get_passage_augmentation(self, path="castorini/msmarco_v1_passage_doc2query-t5_expansions"):
        """
        Loads passage augmentation data and maps document IDs to reformulated queries.
        """
        try:
            dataset = load_dataset(path)
            passage_expansions_dict = {item["id"]: item["predicted_queries"] for item in dataset["train"]}
            return passage_expansions_dict
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return {}

    def eval_solution(self, query_passage_predicted):
        """
        Placeholder function to evaluate a solution.
        """
        return "Evaluation logic not implemented yet"
