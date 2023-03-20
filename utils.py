import requests
from owlready2 import *
import inflect
engine = inflect.engine()

from textblob import TextBlob
import csv
from nltk.corpus import wordnet as wn
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet_ic
import nltk

import numpy as np
import re
from itertools import chain

brown_ic = wordnet_ic.ic('ic-brown.dat')

valid_words = set(nltk.corpus.words.words()) - set(['g', 'l'])

c_str, c_val = zip(*[(k.lower(), float(v)) for k, v in csv.reader(open('food.csv', 'r'))])
categories = dict(zip(c_str, c_val))
category_patterns = [(x, re.compile(f'(^| )(({x})|({engine.plural(x)}))( |$)')) for x in c_str]

# print(category_patterns)

food_ontology = get_ontology("https://raw.githubusercontent.com/FoodOntology/foodon/master/foodon.owl").load()
obo = get_namespace("http://purl.obolibrary.org/obo/")

request_formats = {'RECIPE':{'link':'https://api.edamam.com/api/recipes/v2?type=public&app_id=d55b0c28&app_key=483e0c5071c4a80ac31fad17b1918e6b', 'param': 'q'}, 
'FOOD':{'link':'https://api.edamam.com/api/food-database/v2/parser?app_id=f62e12b4&app_key=42b014ebe865c54314aecc08397faaf5', 'param': 'ingr'}}

from fuzzywuzzy import fuzz, process

def search(name, rtype='RECIPE'):
    return requests.get(request_formats[rtype]['link'], {request_formats[rtype]['param']: name}).json()
    
def get_ingredients(name):
    res = search(name, 'RECIPE')
    for food in res['hits']:
        yield (food['recipe']['ingredients'], food['recipe']['totalWeight'])

def parse_ingredients(ingredients):
    temp = defaultdict(int)
    s = 0
    for ingredient in ingredients:
        # print('INGREDIENT', ingredient['food'])
        # print('CLASSIFIED AS', classify_ingredient(ingredient['food']))
        if (z := classify_ingredient(ingredient['food'])):
            # print('WEIGHT',ingredient['weight'])
            temp[z] += ingredient['weight']
            s += ingredient['weight']
    return dict([(k, v) for k, v in temp.items() if v/s > 0.05])

def ontology_lca(c1, c2):
    return max(c1.ancestors() & c2.ancestors(), key=lambda x: len(x.ancestors()))

def wuf_similarity(c1, c2):
    return 2*len(ontology_lca(c1, c2).ancestors())/(len(c1.ancestors()) + len(c2.ancestors()))

def get_broadest(term, topn=-1):
    return sorted([x for x in food_ontology.search(label=f'*{term}*') if 'obsolete' not in x.label[0]], key=lambda x: len(x.ancestors()))[:topn]

def distance_matrix(concepts, metric=wuf_similarity):
    return [[metric(concepts[x][-1], concepts[y][-1]) for y in range(x)] for x in range(len(concepts))]

def cluster_elements(concepts, start=True):
    if len(concepts) <= 2:
        return [concepts]
    if start:
        concepts = [[x, x] for x in concepts]
    clusters = []
    cur_concepts = concepts[::]
    print(len(concepts))
    matrix = distance_matrix(concepts)
    matrix_backup = [x[::] for x in matrix]
    while matrix:
        if len(matrix) == 1:
            clusters[max(range(len(clusters)), key=lambda x: wuf_similarity(cur_concepts[0][1], clusters[x][1]))][0].append(cur_concepts[0][0])
            break
        x, y = max([(x, y) for x in range(len(matrix)) for y in range(x)], key=lambda x: matrix[x[0]][x[1]])
        clusters.append([[cur_concepts[x][0], cur_concepts[y][0]], ontology_lca(cur_concepts[x][-1], cur_concepts[y][-1])])
        [matrix[i].pop(x) for i in range(x+1, len(matrix))]
        [matrix[i].pop(y) for i in range(y+1, len(matrix))]
        matrix.pop(x)
        matrix.pop(y)
        cur_concepts.pop(x)
        cur_concepts.pop(y)
    return cluster_elements(clusters, start=False)

def path_to_root(concept):
    l = []
    while concept:
        l.append(concept)
        if not concept.is_a:
            break
        concept = concept.is_a[0]
    return l

def triangular_indices(I):
    helper = lambda x: (np.sqrt(2*x + 1/4) - 1/2).astype(int)
    return (z := helper(I)), I - z*(z + 1)/2
    
def get_label(concept):
    try:
        return concept.label[0] if concept.label else ''
    except:
        return ''

# def remove_nonwords(s):
#     return ' '.join([x for x in s.split() if x in valid_words])

def singular_form(s):
    return ' '.join([engine.singular_noun(x) or x for x in s.split()])

from ordered_set import OrderedSet

ANCESTOR_MULTIPLIER = 0.7
def classify_ingredient(name1, verbose=False):
    # name = name.lower()
    for name in OrderedSet([name1, name1.lower()]):
        ls = OrderedSet([name, (z := singular_form(name))] + [str(word) for (word, tag) in TextBlob(z).tags if 'NN' in tag])
        for l in list(ls) + [f'{a}{x}{b}' for x in ls for a in ['', '* '] for b in ['', ' *']]:
            if verbose:
                print(l)
            if not l:
                continue
            if (z := default_world.search(label=l)):
                for cur_class in z:
                    if verbose:
                        print('FOUND LABEL: ', l, get_label(cur_class))
                    ancestors = [cur_class] + list(cur_class.ancestors())
                    ancestors = sorted(ancestors, key=lambda x: -len(x.ancestors()))
                    def scorer(pat):
                        z = 1
                        s = 0
                        for ancestor in ancestors:
                            s += len(pat.findall(get_label(ancestor)))*z
                            z *= ANCESTOR_MULTIPLIER
                        return s
                    scores = [(scorer(pat), cat) for cat, pat in category_patterns]
                    score, cat = max(scores)
                    # print(sorted([(scorer(pat), cat) for cat, pat in category_patterns], reverse=True))
                    if not (ss := sum(list(zip(*scores))[0])):
                        continue
                    if score > 0.1 and score/ss > 0.6:
                        return cat
    else:
        return None
    
def carbon_footprint(recipe):
    weights = parse_ingredients(recipe)
    print(weights)
    return sum([categories[k]*v/1000 for k, v in weights.items()])/sum(weights.values())

# print(default_world.search(label='*apple*')[-1].label)   
# print(engine.plural('tomato'))
# print(classify_ingredient('all-purpose flour', verbose=True))