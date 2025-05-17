import numpy as np
import time
import os
import sys
import csv
import json
import math
from pyrwr.rwr import RWR

"""
Running this file will load in the cropped knowledge graph - assertion list and use the RWR implementation
(https://github.com/jinhongjung/pyrwr) to determine the semantic consistency matrix. The resulting matrix will be stored
in dictionary format in a JSON file.
"""

# select paths of the lookup table and cropped knowledge graph
path_project = os.path.abspath(os.path.join(__file__, "../.."))
lookup_filepath = os.path.join(path_project, "Semantic Consistency/KG_lookup_55.csv")
cropped_filepath = os.path.join(path_project, "Semantic Consistency/KG_crop_55.csv")
save_filepath = os.path.join(path_project, "Semantic Consistency/Stored matrices/my_dataset_kg_55_info.json")


c = 0.15            # Restart probability
epsilon = 1e-9      # Error tolerance for power iteration
max_iters = 1000    # Maximum number of iterations for power iteration


class NumpyEncoder(json.JSONEncoder):
    """
    Convert Numpy arrays to JSON
    Source: https://pynative.com/python-serialize-numpy-ndarray-into-json/
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


np.set_printoptions(threshold=sys.maxsize)  # don't truncate printing
start = time.time()

input_graph = []

# Load in the relevant assertions from the csv file and convert to the right format.
with open(cropped_filepath, newline='', encoding="utf8") as csvfile:
    spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
    counter = 0
    for row in spamreader:
        counter = counter + 1
        c1idx = row[4]
        c2idx = row[5]
        w = row[3]
        entry = ('{} \t {} \t {}'.format(c1idx, c2idx, w))
        input_graph.append(entry)
        if counter % 100000 == 0:
            print(counter)

print("length of the input graph: ", len(input_graph))
print(input_graph[0])

concept_table = []

# Load in the concept-integer conversion table from the csv file.
with open(lookup_filepath, newline='', encoding="utf8") as csvfile:
    spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
    counter = 0
    for row in spamreader:
        counter = counter + 1
        concept = row[0]
        number = row[1]
        entry = (concept, number)
        concept_table.append(entry)
        if counter % 100000 == 0:
            print(counter)

print(concept_table[0])
# The 91 COCO concepts (all 20 VOC concepts are included).
interesting_concepts = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                        'traffic_light', 'fire_hydrant', 'street_sign', 'stop_sign', 'parking_meter', 'bench',
                        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                        'hat', 'backpack', 'umbrella', 'shoe', 'eyeglasses', 'handbag', 'tie', 'suitcase',
                        'frisbee', 'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat', 'baseball_glove',
                        'skateboard', 'surfboard', 'tennis_racket', 'bottle', 'plate', 'wine_glass', 'cup',
                        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
                        'carrot', 'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted_plant', 'bed',
                        'mirror', 'dining_table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse',
                        'remote', 'keyboard', 'cell_phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
                        'blender', 'book', 'clock', 'vase', 'scissors', 'teddy_bear', 'hair_dryer', 'toothbrush',
                        'hair_brush'
                        )

list_of_relevant_indexes = []

# From the conversion table, find which integers belong to the concepts.
for i in range(len(interesting_concepts)):
    concept = interesting_concepts[i]
    relevant_index = [x for x in concept_table if x[0] == concept]
    list_of_relevant_indexes.append(relevant_index)



s = np.zeros((len(list_of_relevant_indexes), len(list_of_relevant_indexes)))

# Compute the random walk with restart score vector r for each COCO concept as a seed, and extract the scores related
# to the other COCO concepts to get a 91x91 matrix S.
for i in range(len(list_of_relevant_indexes)):
    seed = int(list_of_relevant_indexes[i][0][1])
    rwr = RWR()
    rwr.read_graph(input_graph, graph_type='directed')
    r = rwr.compute(seed, c, epsilon, max_iters)

    for j in range(len(list_of_relevant_indexes)):
        c_nr = int(list_of_relevant_indexes[j][0][1])
        s[i, j] = r[c_nr]

# Compute a symmetrical matrix based on (l1,l2) = (l2,l1) = sqrt(l1*l2).
s_sym = np.zeros((len(list_of_relevant_indexes), len(list_of_relevant_indexes)))

for i in range(len(list_of_relevant_indexes)):
    for j in range(len(list_of_relevant_indexes)):
        s_sym[i, j] = math.sqrt(s[i, j] * s[j, i])

print("S_sym: ", s_sym)

# Store the matrix in a dictionary, so we can store it in a JSON later.
KG_COCO_info = {}
KG_COCO_info['S'] = s_sym

s_sym_voc = np.zeros((6, 6))

# Select the sequence numbers of categories within the 'interesting_concepts' list in your own dataset
coco_to_mydataset_voc = [6, 3, 2, 1, 8, 4]

# containing only relevant VOC data.
for l1 in range(0, 6):
    for l2 in range(0, 6):
        s_sym_voc[l1, l2] = s_sym[coco_to_mydataset_voc[l1]-1, coco_to_mydataset_voc[l2]-1]

KG_My_DataSet_info = {}
KG_My_DataSet_info['S'] = s_sym_voc

# Store all dictionaries in another dictionary.
info = {}
info['KG_My_DataSet_info'] = KG_My_DataSet_info

# Save all the info to a JSON file.
with open(save_filepath, 'w') as j:
    json.dump(info, j, cls=NumpyEncoder)

end = time.time()
hours, rem = divmod(end - start, 3600)
minutes, seconds = divmod(rem, 60)
print("time it took to determine S: ")
print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))