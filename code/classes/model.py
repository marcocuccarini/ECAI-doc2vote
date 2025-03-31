from rank_bm25 import BM25Okapi


#classe che implementa il funzionamento del modell BM25

class BM25:

    def __init__(self, passages):

        self.tokenized_corpus = [doc.split(" ") for doc in passages]

        self.bm25 = BM25Okapi(self.tokenized_corpus)


    def ranking_score(self, query, passages, top_n="10"):

        ranking_score={}

        ranking_passages={}
        

        for id_query, text_query in query.items():

            tokenized_query = text_query.split(" ")

            vet_score, vet_passage = zip(*sorted(zip(self.bm25.get_scores(tokenized_query), passages.keys()), reverse=True))

            ranking_score[id_query] = vet_score[:top_n]

            ranking_passages[id_query] = vet_passage[:top_n]

        
        return ranking_score, ranking_passages      

    def ranking_score_partial(self, query, passages, index, top_n="10"):

        ranking_score={}

        ranking_passages={}
        

        for id_query, text_query in query.items():

            tokenized_query = text_query.split(" ")

            vet_score, vet_passage = zip(*sorted(zip(self.bm25.get_scores(tokenized_query), index), reverse=True))

            ranking_score[id_query] = vet_score[:top_n]

            ranking_passages[id_query] = vet_passage[:top_n]

        
        return ranking_score, ranking_passages      



    def ranking_passages(self, query, passages, top_n):

        id_passage=list(passages.keys())

        text_passages=list(passages.values())

        ranking_passages={}
        cont=0

        doc_ids = list(query.keys())

        for id_query, text_query in query.items():

            tokenized_query = text_query.split(" ")

            ranking_passages[id_query]=self.bm25.get_top_n(tokenized_query, id_passage, n=top_n)



        return ranking_passages


class BERT_Model:

    def __init__(self, model_name, passage, SentenceTransformer):


        self.model_name = model_name

        self.model=SentenceTransformer(self.model_name)

        self.passage=passage

        self.passage_encoded=self.text_embedding(passage)



    def text_embedding(self, text):

        return self.model.encode(text, convert_to_tensor=False)

    def similarity_extraction(user_arg_enc, KB_enc):

        KB_sim=[]

        # to normilize the similarity I consider the value tha assume when the are two equal texts

        norm_term=np.dot(user_arg_enc,user_arg_enc)


        for i in range(len(KB_enc)):

            temp=[]

            for j in range(len(KB_enc[i])):

                #calcolo la similarity con il dot product normalizzandolo

                temp.append(np.dot(user_arg_enc,KB_enc[i][j])/norm_term)

            KB_sim.append(temp) 

        return KB_sim


    def ranking_score(self, query):

        query_encoded=self.model.encode(query, convert_to_tensor=False)
        return similarity_extraction(query_encoded, self.passage_encoded)


    def ranking_passages(self, query, passages, top_n,):


        sim_matrix = ranking_score(query)

        sim_sorted, sort_passage = zip(*sorted(zip(sim_matrix, passages), reverse=True))


        return sort_passage[:top_n]



















