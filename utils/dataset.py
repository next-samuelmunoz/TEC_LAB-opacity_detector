# -*- coding: utf-8 -*-

import json


def load(path_dataset):
    '''Load Dataset
    '''
    with open(path_dataset+'/data.json', 'r') as fp:
        data = json.load(fp)
    shape_canonical = data['shape_canonical']
    imgs = data['imgs']
    for img in imgs:
        img['path'] = path_dataset+'/'+img['path']
    return data
