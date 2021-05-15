import numpy as np
import torch
from bson import ObjectId
import config
import requests

def create_output_start_action():
    action = np.zeros((1, config.output_size))
    return action[config.seq_types.index('start')]

def normalize_url(url):
    parts = url.split("/")
    parts = [part if (not ObjectId.is_valid(part)) else 'objectId' for part in parts]
    return "/".join(parts)

def getObjectType(normalized_url, url):
    urls = [
        '/api/links/from/objectId',
        '/api/links/either/objectId',
        '/api/objects/objectId',
        '/api/objects/objectId/objectId',
        '/api/links/to/objectId'
    ]
    if normalized_url in urls: #Find object type
        objectId = url.split("/")[-1]
        r = requests.get(f"http://localhost:9000/api/objects/type/{objectId}")
        try:
            return r.json()['type']
        except:
            print ("***********Error ", url, normalized_url)
            return ''
    if normalized_url == '/api/contributions/objectId':
        pass #this is a post request, need info from post data
    if normalized_url == '/api/links/objectId':
        objectId = url.split("/")[-1]
        r = requests.get(f"http://localhost:9000/api/links/type/{objectId}")
        try:
            return r.json()['type']
        except:
            print ("***********Error ", url)
            return ''
    return ''

def create_input_action(req_type, url, delta, objType):
    x = np.zeros(config.input_size)
    x[:4] = config.type_to_cat[req_type]
    x[4+config.urls.index(url)] = 1
    x[-2] = delta
    x[-1] = config.objType_to_id[objType]
    return x

def parse_log_line(line):
    parts = line.split()
    req_type = parts[6][1:]
    user = parts[1].split(":")[-1]
    url = parts[7]
    time = parts[4][-8:]

    return req_type, user, url, time


def parse_log_file(filename):
    lines = []
    urls = []
    req_types = []
    lines.append(create_input_action('GET', 'start', 0.0, ''))
    with open(filename, 'r') as f:
        line = f.readline()
        req_type, user, url, time = parse_log_line(line)
        prev_secs = int(time[-2:])
        norm_url = normalize_url(url)
        objectType = getObjectType(norm_url, url)
        urls.append(norm_url)
        req_types.append(req_type)
        lines.append(create_input_action(req_type, norm_url, 0, objectType))
        for line in f:
            req_type, user, url, time = parse_log_line(line)
            secs = int(time[-2:])
            delta = secs - prev_secs
            if req_type in config.cat_to_request_type and (delta > 1 or req_type != 'GET'): #TODO validate url
                print(req_type, url)
                norm_url = normalize_url(url)
                objectType = getObjectType(norm_url, url)
                urls.append(norm_url)
                lines.append(create_input_action(req_type, norm_url, delta, objectType))
                req_types.append(req_type)
                prev_secs = secs

    lines.append(create_input_action('GET', 'end', 0.0, ''))
    return np.array(lines), urls, req_types
