import numpy as np

def clean_data(x):
    """Met en lowercase et supprime les espaces dans une string ou liste"""
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    elif isinstance(x, str):
        return str.lower(x.replace(" ", ""))
    return ''

def create_soup(x):
    """Crée une 'soupe' de mots à partir des colonnes clés"""
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

def get_director(crew):
    """Retourne le nom du réalisateur"""
    for member in crew:
        if member['job'] == 'Director':
            return member['name']
    return ''

def get_list(x):
    """Retourne les 3 premiers éléments d'une liste de dictionnaires"""
    if isinstance(x, list):
        names = [i['name'] for i in x]
        return names[:3]
    return []
