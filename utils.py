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

brown_ic = wordnet_ic.ic('ic-brown.dat')

valid_words = set(nltk.corpus.words.words()) - set(['g', 'l', 'water'])

c_str, c_val = zip(*[(k.lower(), float(v)) for k, v in csv.reader(open('food.csv', 'r'))])
categories = dict(zip(c_str, c_val))

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
            temp[z] += ingredient['weight']
            s += ingredient['weight']
    return dict([(k, v) for k, v in temp.items() if v/s > 0.05])


def classify_ingredient(name, verbose=False):
    name = name.lower()
    ls = [name] + [y for (word, tag) in TextBlob(name).tags if 'NN' in tag for y in [str(word), engine.singular_noun(word)] if y]
    # print(ls)
    for l in ls + [f'{a}{x}{b}' for x in ls for a in ['', '* '] for b in ['', ' *']]:
        if verbose:
            print(l)
        if not l:
            continue
        if (z := default_world.search(label=l)):
            for cur_class in z:
                try:
                    if verbose:
                        print('FOUND LABEL: ', l, cur_class.label)
                    while True:
                        if verbose:
                            print(cur_class.label)
                        if not cur_class.label:
                            break
                        label = ' '.join([word for word in cur_class.label[0].lower().split() if word in valid_words])
                        if not label or any([x in label for x in ['European','entity']]):
                            break
                        Z = process.extract(cur_class.label[0], c_str, limit=1, scorer=fuzz.partial_ratio)
                        if verbose:
                            print(Z)
                        if Z and Z[0][1] > 90:
                            return Z[0][0]
                        # if (z := max([(sentence_similarity(category, label), category) for category in c_str]))[0] > 0.8:
                        #     print(z)
                        #     return z[1]

                        if cur_class.is_a:
                            cur_class = cur_class.is_a[0]
                        else:
                            if verbose:
                                print('TERMINAL')
                            break
                except Exception as e:
                    if verbose:
                        print(e)
                    continue
    else:
        return None
    
def carbon_footprint(recipe):
    weights = parse_ingredients(recipe)
    print(weights)
    return sum([categories[k]*v/1000 for k, v in weights.items()])/sum(weights.values())

# print(default_world.search(label='*apple*')[-1].label)   