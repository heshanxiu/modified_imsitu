import os
import numpy
import torch
import json
from sklearn.model_selection import train_test_split

root = "/local/shanxiu/lxmert/imSitu/visualsrl_data_dev"
file_name = "/local/shanxiu/lxmert/snap/visualsrl/visualsrl_extract_dev_15656/image_id_to_index.json"

all_data = json.load(open(file_name))
index_list = list(all_data.keys())

train, test = train_test_split(index_list, test_size=0.2)
true_train, dev = train_test_split(train, test_size=0.25)


with open(os.path.join(root, "train.json"), "w") as fp:
    for i in true_train:
        fp.write(f'{i}\n')
with open(os.path.join(root, "test.json"), "w") as fp:
    for i in test:
        fp.write(f'{i}\n')

with open(os.path.join(root, "dev.json"), "w") as fp:
    for i in dev:
        fp.write(f'{i}\n')



# arr = []
# with open("filename", "r") as fp:
#     arr = [id for id in fp.readlines()]