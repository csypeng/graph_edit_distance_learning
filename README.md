# graph_edit_distance_learning

## IJCAI2021 paper

Graph Edit Distance Learning via Modeling Optimum Matchings with Constraints

## Data preparation

### Compute associate graph

Given two graphs G1 and G2, you need to compute the associate graph of them. The idea is that for each node v1 in G1 and each node u1 in G2, a node (v1,u1) is added into the associate graph. If v1's label is equal to u1's label, the label of (v1,u1) is 0; otherwise, the label of (v1,u1) is 1. For each edge (v1,v2) in G1, if (u1,u2) is not in G2, (v1,u1) and (v2,u2) has an edge in the associate graph. For each edge (u1,u2) in G2, if (v1,v2) is not in G1, (v1,u1) and (v2,u2) has an edge in the associate graph. In addition, for each sumToOne constraint, a node is added to the associate graph. 


### Compute ground-truth node matchings

For each pair of graphs, the GED of them and one ground-truth node matching is needed. You can use the existing exact GED computation methods to compute the ground-truth node matchings, e.g., https://github.com/LijunChang/Graph_Edit_Distance and https://github.com/JongikKim/Inves

An example of the node matching file is as follows.

GED 22 nodeMapNum 1

0|5 1|4 2|3 3|2 4|1 5|0 6|6 7|7 8|9 9|13 10|14 -1|8 -1|10 -1|11 -1|12 -1|15 -1|16 -1|17 -1|18 -1|19 -1|20

In this example, the GED of the two graphs is 22. 0|5 means node0 of G1 is matched to node5 of G2. -1|10 means node10 of G2 is deleted.


## Run the model

Install the required packages, including 

* DGL (dgl 0.5.3, dgl-cu110 0.5.3)
* pytorch 1.6.0
* numpy 1.16.0
* networkx 2.5


Setup the parameters and file address in main.py and Dataset.py 

* "dirName" in main.py is the address of associate graphs
* "ged_node_map_file_addr" in Dataset.py is the address of node matchings

python main.py to run the model


