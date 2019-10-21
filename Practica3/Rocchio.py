"""
.. module:: SearchIndexWeight
SearchIndex
*************
:Description: Rocchio
    Performs a AND query for a list of words (--query) in the documents of an index (--index)
    You can use word^number to change the importance of a word in the match
    --nhits changes the number of documents to retrieve
:Authors: zhang & pan
    
:Version: 
:Created on: 21/10/2019 14:17 
"""

from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError

import argparse

from elasticsearch_dsl import Search
from elasticsearch_dsl.query import Q

__author__ = 'pan & zhang'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', default=None, help='Index to search')
    parser.add_argument('--nhits', default=10, type=int, help='Number of hits to return')
    parser.add_argument('--query', default=None, nargs=argparse.REMAINDER, help='List of words to search')

    args = parser.parse_args()

    index = args.index
    query = args.query
    print(query)
    nhits = args.nhits

    try:
        client = Elasticsearch()
        s = Search(using=client, index=index)

        if query is not None:
            q = Q('query_string',query=query[0])
            for i in range(1, len(query)):
                q &= Q('query_string',query=query[i])

            s = s.query(q)
            response = s[0:nhits].execute()
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
