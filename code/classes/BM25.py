from rank_bm25 import BM25Okapi

class BM25:

    def __init__(self, passages):

        self.tokenized_corpus = [doc.split(" ") for doc in passages]

        self.bm25 = BM25Okapi(self.tokenized_corpus)


    def ranking_score(self, query):

        ranking_score={}

        for id_query, text_query in query.items():

            tokenized_query = query.split(" ")

            ranking_score[id_query]=self.bm25.get_scores(tokenized_query)

        return ranking_score      

    def ranking_passages(self, query, passages, top_n, real_answer):

        id_passage=list(passages.keys())
        text_passages=list(passages.values())[:50000]

        ranking_passages={}
        cont=0

        doc_ids = list(query.keys())



        for id_query, text_query in query.items():

            tokenized_query = text_query.split(" ")

            ranking_passages[id_query]=self.bm25.get_top_n(tokenized_query, text_passages, n=top_n)

            #print("query-->",text_query)

            #print("rank-->",ranking_passages[id_query][0])

            #print("real_answer-->",passages[real_answer[id_query][0]])


            '''for i in range(len(ranking_passages[id_query])):

                if(ranking_passages[id_query][i] == passages[real_answer[id_query][0]]):



                    print("Match-done")
                print(ranking_passages[id_query][i],passages[real_answer[id_query][0]])


                print("----------------")




            cont+=1

            if (cont==10):

                break'''


        return ranking_passages
