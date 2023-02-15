import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial
import time
import json

import subprocess
from subprocess import PIPE, Popen
import os
from tqdm import tqdm

def worker(entity_name, i,num=10):
    entity_name = entity_name.replace(" ","_").replace(':', '').replace('*', '')
    search_keyword = entity_name

    photo_dir = "./all_dbphoto/" + entity_name
    if not os.path.exists(photo_dir):
        os.mkdir(photo_dir)

    command = "python image_downloader.py --engine Google --proxy_http 127.0.0.1:4780 --driver chrome_headless --max-number {} --num-threads 48 --timeout 5 --output ./all_dbphoto/".format(num) + entity_name + " " + '''"''' + search_keyword.replace("_"," ") + '''"'''

    os.system(command)


def collectMyResult(result):
    print("Got result {}".format(result))

def abortable_worker(func, *args, **kwargs):
    timeout = 300
    p = ThreadPool(1)
    res = p.apply_async(func, args=args)
    try:
        out = res.get(timeout)  # Wait timeout seconds for func to complete.
        return out
    except multiprocessing.TimeoutError:
        print("Aborting due to timeout")
        raise

if __name__ == "__main__":
    with open("./db_data0309.json","r",encoding='utf-8') as f:
        all_data = json.load(f)

    pool = multiprocessing.Pool(processes=1,maxtasksperchild=1)

    featureClass = [[all_data[i],i] for i in tqdm(range(0,len(all_data),1))] #list of arguments
    for f in featureClass:
        abortable_func = partial(abortable_worker, worker, timeout=300)
        pool.apply_async(abortable_func, args=f,callback=collectMyResult)
    pool.close()
    pool.join()
