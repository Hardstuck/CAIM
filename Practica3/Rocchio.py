"""
.. module:: SearchIndexWeight

SearchIndex
*************

:Description: SearchIndexWeight

    Performs a AND query for a list of words (--query) in the documents of an index (--index)
    You can use word^number to change the importance of a word in the match

    --nhits changes the number of documents to retrieve

:Authors: bejar
    

:Version: 

:Created on: 04/07/2017 10:56 

"""

from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError
from elasticsearch.client import CatClient

import argparse

import numpy as np
from elasticsearch_dsl import Search
from elasticsearch_dsl.query import Q
import operator


__author__ = 'bejar'

def doc_count(client, index):
    return int(CatClient(client).count(index=[index], format='json')[0]['count'])


def document_term_vector(client, index, id):

    termvector = client.termvectors(index=index, id=id, fields=['text'],
                                    positions=False, term_statistics=True)

    file_td = {}
    file_df = {}

    if 'text' in termvector['term_vectors']:
        for t in termvector['term_vectors']['text']['terms']:
            file_td[t] = termvector['term_vectors']['text']['terms'][t]['term_freq']
            file_df[t] = termvector['term_vectors']['text']['terms'][t]['doc_freq']
    return sorted(file_td.items()), sorted(file_df.items())


def toTFIDF(client, index, file_id):
    # Get document terms frequency and overall terms document frequency
    file_tv, file_df = document_term_vector(client, index, file_id)
    
    max_freq = max([f for _, f in file_tv])

    dcount = doc_count(client, index)
    
    tfidfw = {}
    
    for (t, w),(_, df) in zip(file_tv, file_df):
        tfidfw[t] = ((w/max_freq)*(np.log(dcount/df)))
        
    return tfidfw #dictionary of terms and weights


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', default=None, help='Index to search')
    parser.add_argument('--nhits', default=10, type=int, help='Number of hits to return')
    parser.add_argument('--query', default=None, nargs=argparse.REMAINDER, help='List of words to search')

    args = parser.parse_args()
    
    #parameters
    nrounds = 5
    alpha = 0.95
    beta = 0.05
    R = 3

    index = args.index
    query = args.query
    print(query)
    nhits = args.nhits

    try:
        client = Elasticsearch()
        s = Search(using=client, index=index)

        if query is not None:
            for i in range(0,nrounds):
                #busquem els nhits documents més rellevants
                q = Q('query_string',query=query[0])
                for i in range(1, len(query)):
                    q &= Q('query_string',query=query[i])

                s = s.query(q)
                response = s[0:nhits].execute()
                #######################################
                query_dict = {}
            
                for q in query:
                    if('^' in q):
                        key, value = q.split('^')
                        query_dict[key] = float(value)
                    else:
                        query_dict[q] = 1.0
                        
                print(query_dict)
                
                sumDocs = {}
                
                    #per cada document calcular el seu TFIDF
                for r in response:
                    #getting the weights of every document
                    file_tw = toTFIDF(client, index, r.meta.id)
                    #sumDocs(t) = actualweight(t) + termweightinD_i(t)
                    sumDocs = {t: sumDocs.get(t,0) + file_tw.get(t,0) for t in set(file_tw) | set(sumDocs)}
                        
                #compute beta*(d_1 + ... + d_2)/k
                sumDocs = {t: beta*sumDocs.get(t,0)/nhits for t in set(sumDocs)} #beta * vector de documents / K
                oldQuery = {t: alpha*query_dict.get(t,0) for t in set(query_dict)} #alpha * query
                newQuery = {t: sumDocs.get(t,0) + oldQuery.get(t,0) for t in set(oldQuery)|set(sumDocs)} #newquery = sumDocs + oldquery
                
                newQuery = sorted(newQuery.items(), key=operator.itemgetter(1), reverse=True)
                newQuery = newQuery[:R]
                
                query = []
                for (term, value) in newQuery:
                    query.append(term + '^' + str(value))
                
                print(query)
                
                
            for r in response:  # only returns a specific number of results
                print(f'ID= {r.meta.id} SCORE={r.meta.score}')
                print(f'PATH= {r.path}')
                print(f'TEXT: {r.text[:50]}')
                print('-----------------------------------------------------------------')

        else:
            print('No query parameters passed')

        print (f"{response.hits.total['value']} Documents")

    except NotFoundError:
        print(f'Index {index} does not exists')

