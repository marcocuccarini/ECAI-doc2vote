

class VotePipeline:


    def __init__(self, model, dataset):

        self.model=model

        self.dataset=dataset


    def augmented_passages(self, dataset.passage, dataset.passage_augmented):

        #crea una lista di tutti i passaggi augmentati

        list_passages_augmented=[]
        index_passages_augmented=[]

        for i in dataset.passage.keys():

            #aggiungo alla lista lìelemento originale

            list_passages_augmented.append(dataset.passage[i])
            index_passages_augmented.append(i)

            for j in dataset.passage_augmented[i]:

                #aggiungo agni singoli sentence augmented

                list_passages_augmented.append(dataset.passage[j])
                index_passages_augmented.append(i)


        return list_passages_augmented, index_passages_augmented



    def vote2vec(self):

        #prima cosa devo ottenere la lista completa di valori. 

        list_passages_augmented, index_passages_augmented = augmented_passage(dataset.passage, dataset.passage_augmented)

        ranking_passages = BM25.ranking_score(dataset.query)

        dict_ranking_index={}

        for i in ranking_passages.keys():

            ranking_passage, ranking_index = zip(*sorted(zip(ranking_passages[i], index_passages_augmented[i]), reverse=True))

            dict_ranking_index[i] = ranking_index


        return dict_ranking_index


















