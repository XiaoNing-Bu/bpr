import lmdb
import json
import os
import shutil
import csv
if os.path.exists('lmdb'):
    shutil.rmtree('lmdb')
def write_to_lmdb(db, key, value):
    """
    Write (key,value) to db
    """
    success = False
    while not success:
        txn = db.begin(write=True)
        try:
            txn.put(str(key).encode("utf-8"), json.dumps(value).encode("utf-8"))
            txn.commit()
            success = True
        except lmdb.MapFullError:
            txn.abort()
            # double the map_size
            curr_limit = db.info()['map_size']
            new_limit = curr_limit*2
            print('>>> Doubling LMDB map size to %sMB ...' % (new_limit>>20,))
            db.set_mapsize(new_limit)

db = lmdb.open('lmdb', map_size=int(1e10))
ids = []
values = []
with open("psgs_w100.tsv") as i_f:
    tsv_reader = csv.reader(i_f, delimiter="\t")
    i=0
    for row in tsv_reader:
        if row[0] == "id":
            continue  # ignoring header
        _id, text, title = int(row[0]), row[1], row[2]
        write_to_lmdb(db, _id, (title, text))