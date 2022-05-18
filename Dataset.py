
import os
import numpy as np
from dgl.data import DGLDataset
import dgl
import torch
import networkx as nx

GPUID=2

def read_and_split_to_individual_graph(fname, gsizeNoLessThan=15, gsizeLessThan=30):
    '''
    :parm fname: the file storing all graphs in the format as follows:
        t # gid 
        v 1 0
        v 2 0
        e 1 2 1
    '''

    f = open(fname)
    print(fname)

    lines = f.read()
    f.close()

    lines2 = lines.split("t # ")

    lines3 = [g.strip().split("\n") for g in lines2]

    glist = []
    for idx in range(1, len(lines3)):
        if len(glist) >= 420: 
            break

        cur_g = lines3[idx]
        
        gid_line = cur_g[0].strip().split(' ')       
        gid = gid_line[0]
        if len(gid_line) == 6:
            glabel = gid_line[3] # ged
            
            # on IMDB and PTC, such pair of graphs will make Cuda out-of-memory
            if float(glabel) > 395.0: #395.0 for IMDB, 80 for PTC
                continue
            
            leftGNodeNum = int(gid_line[4])
            rightGNodeNum = int(gid_line[5])
            g = nx.Graph(id = gid, label = glabel, leftGsize=leftGNodeNum, rightGsize=rightGNodeNum)
        else:
            print("need to give GED as the label of a prod graph")
            exit(-1)

        
        for idy in range(1, len(cur_g)):
            tmp = cur_g[idy].split(' ')
            if tmp[0] == 'v':
                g.add_node(tmp[1], att=int(tmp[2]))
            if tmp[0] == 'e':
                g.add_edge(tmp[1], tmp[2], att=int(tmp[3]))
        

        if g.number_of_nodes() >= gsizeNoLessThan and g.number_of_nodes() < gsizeLessThan:          
            glist.append(g)
    
    
    return glist
    




class GINDataset(DGLDataset):
    
    graphList = None

    def __init__(self, name, self_loop, degree_as_nlabel=False,
                 raw_dir=None, start=None, end=None, force_reload=False, verbose=False,
                 num_predNodeMap=1):

        self._name = name  
        gin_url = ""
        self.ds_name = ''
        self.file = raw_dir
        self.start = start
        self.end = end
        self.num_predNodeMap = num_predNodeMap

        self.self_loop = self_loop
        self.graphs = []
        self.labels = []
        self.ground_truth_nodeMaps = [] # each graph just picks one gt nodeMap
        self.all_ground_truth_nodeMaps = [] # each graph stores all its gt nodeMaps
        self.gids = [] # id of each graph

        # relabel
        self.glabel_dict = {}
        self.nlabel_dict = {}
        self.elabel_dict = {}
        self.ndegree_dict = {}

        # global num
        self.N = 0  # total graphs number
        self.n = 0  # total nodes number
        self.m = 0  # total edges number

        # global num of classes
        self.gclasses = 0
        self.nclasses = 0
        self.eclasses = 0
        self.dim_nfeats = 0

        # flags
        self.degree_as_nlabel = degree_as_nlabel
        self.nattrs_flag = False
        self.nlabels_flag = False

        super(GINDataset, self).__init__(name=name, url=gin_url, hash_key=(name, self_loop, degree_as_nlabel),
                                         raw_dir=raw_dir, force_reload=force_reload, verbose=verbose)

    @property
    def raw_path(self):
        return os.path.join(".", self.raw_dir)


    def download(self):
        pass


    def __len__(self):
        return len(self.graphs)



    def __getitem__(self, idx):
        if idx < len(self.graphs):
            return self.graphs[idx], self.labels[idx], self.ground_truth_nodeMaps[idx], None, self.gids[idx]
        else:
            return None, None


    def _file_path(self):
        return self.file


    def process(self):
        """ Loads input dataset from dataset/NAME/NAME.txt file
        """
       
        if self.verbose:
            print('loading data...')

        with open(self.file, 'r') as f:
            print("read graphs from ", self.file)
            if GINDataset.graphList is None:
                print("graphList is None..............")
                GINDataset.graphList = read_and_split_to_individual_graph(self.file, gsizeNoLessThan=0, gsizeLessThan=10000000)  
            else:
                print("graphList is not None.............")

            glist = GINDataset.graphList[self.start:self.end]

            # ged_node_map_file_addr = "node_matching_examples/nodeMap2000/" # for AIDS
            ged_node_map_file_addr = "../Inves/IMDBnodeMap2000/" # for IMDB
            # ged_node_map_file_addr = "node_matching_examples/linuxnoLs30SampleNodeMap2000/" # for Linux
            # ged_node_map_file_addr = "node_matching_examples/PTCnodeMap2000/" # for PTC

            for g in glist:
                gidtmp = g.graph.get("id").split("|")
                smallGID = gidtmp[0]
                largeGID = gidtmp[1]
                smallGNodeNum = g.graph.get("leftGsize")
                largeGNodeNum = g.graph.get("rightGsize")
                nodeMapFname = ged_node_map_file_addr+"g"+smallGID+"g"+largeGID+"nodeMap.txt"
                
                if os.path.isfile(nodeMapFname) == False:
                    print("node map file not exist")
                    continue
                
                f = open(nodeMapFname)
                nodeMaps = f.read()
                f.close()
                nodeMaps = nodeMaps.strip().split('\n')
                nodeMaps = nodeMaps[1:] # ignore the first line
                nodeMap1 = nodeMaps[0]
  
                nodeMap1 = nodeMap1.strip().split(" ")

                positive_part_of_nodeMap1 = set()
                for ele in nodeMap1:
                    abc = ele.split("|")
                    if int(abc[0]) >= 0:
                        positive_part_of_nodeMap1.add(ele)

                # In nodeMap, fromGraph only have ONE -1 node to denote insertion
                # The -1 node of fromGraph can map to many nodes of toGraph
                # For example, -1|toGraphNode1, -1|toGraphNode2, -1|toGraphNode3, ...
                # However, in prodG, fromGraph has |toGraph.size| - |fromGraph.size| negative nodes
                # For example, -1|toGraphNode1, -2|toGraphNode2, -3|toGraphNode3, ...
                # Therefore, sort toGraphNodes in -1|toGraphNode1, -1|toGraphNode2, ... in nodeMap
                # and map them to the -1, -2, -3 ... nodes of fromGraph of prodG

                nodeMap1_new = processNegOneLeftNodeInNodeMap(nodeMap1)
                # print("nodeMap1_new", nodeMap1_new)

                nodeMap1_set = set(nodeMap1_new)
                        
                # add special nodes for loss3
                # loss3 is L_m in paper
                g.add_node("--1", att=-2) # weight=-2 denotes special node for loss3
                

                oldID2newID = {}
                newID2oldID = {}

                count = 0
                for smallGNodeID in range(-(largeGNodeNum - smallGNodeNum), smallGNodeNum):
                    for largeGNodeID in range(0, largeGNodeNum):
                        curNode = str(smallGNodeID)+"|"+str(largeGNodeID)  
                        oldID2newID[curNode] = count
                        newID2oldID[count] = curNode
                        count = count + 1
              
                for node in g.nodes():
                    if '|' not in node:
                        oldID2newID[node] = count
                        newID2oldID[count] = node
                        count = count + 1
                

                nodeMap1_tensor = torch.zeros(largeGNodeNum, largeGNodeNum).cuda()
                
                # just make the positive part of nodeMap1 to 1
                for ele in positive_part_of_nodeMap1:
                    newID_ele = oldID2newID[ele]
                    rowID = int(newID_ele / largeGNodeNum)
                    colID = newID_ele % largeGNodeNum
                    nodeMap1_tensor[rowID, colID-1] = 1.0

                self.ground_truth_nodeMaps.append(nodeMap1_tensor)
                


                edges = [[],[]]
                for edge in g.edges():
                    # note that dgl is directed
                    left = edge[0]
                    right = edge[1]
                    if g.nodes[left]['att'] == -1:
                        # left is a special node
                        # We only have directed edge from non-special node to special node
                        edges[0].append(oldID2newID[edge[1]])
                        edges[1].append(oldID2newID[edge[0]])
                    else:
                        if g.nodes[right]['att'] == -1:
                            # right is a special node
                            edges[0].append(oldID2newID[edge[0]])
                            edges[1].append(oldID2newID[edge[1]])
                        else:
                            edges[0].append(oldID2newID[edge[0]])
                            edges[1].append(oldID2newID[edge[1]])

                            edges[0].append(oldID2newID[edge[1]])
                            edges[1].append(oldID2newID[edge[0]])

                # each special node for loss3 connects to all nodes in the prodG, 
                # but no connection with the special nodes for loss2  
                # how to differentiate the nodes for loss2 and for loss3
                # weight of nodes for loss2 is -1, name of nodes for loss2 is -1, -2, -3, ...
                # weight of nodes for loss3 is -2, name of nodes for loss3 is --1, --2, ...
                for node in g.nodes():
                    firstSpecialNodeForLoss3 = oldID2newID['--1']
                    # secondSpecialNodeForLoss3 = oldID2newID['--2']
                    if g.nodes[node]['att'] != -1 and g.nodes[node]['att'] != -2:
                        # add directed edge from non-special node 
                        edges[0].append(oldID2newID[node])
                        edges[1].append(firstSpecialNodeForLoss3)

                        g.add_edge(node, '--1', att=0) # weight=0 is just a place holder


                dg = dgl.graph((torch.tensor(edges[0]), torch.tensor(edges[1])))

                # set node type and cost 
                not_special_list = [] # if a node is a non-special node
                ntypelist = []
                ncostlist = []
                nidlist = []
                neigh_type_count_list = [] # for a node, the number of neighbors of different types
                special_cost_list = [] # for loss2 
                special_cost_list_for_loss3 = [] # for loss3
                numPredNodeMap = self.num_predNodeMap # number of predicted nodeMaps
                for node in dg.nodes():
                    nidlist.append(node.item())
                    oldID = newID2oldID[node.item()]
                    ncost = g.nodes[oldID]['att']
                    ncostlist.append(ncost)
                    if ncost == 0:
                        ntypelist.append(0)
                        special_cost_list.append(0)
                        special_cost_list_for_loss3.append([0]*numPredNodeMap)
                        not_special_list.append(1.0)
                    if ncost == 1:
                        ntypelist.append(1)
                        special_cost_list.append(0)
                        special_cost_list_for_loss3.append([0]*numPredNodeMap)
                        not_special_list.append(1.0)
                    if ncost == -1: 
                        # ncost == -1 means special node for loss2, i.e., SumOneConstraint
                        # it does not matter we regard the special node as type 1 or type 0
                        # as there is no edge going out from a special node
                        ntypelist.append(0)
                        special_cost_list.append(1)
                        special_cost_list_for_loss3.append([0]*numPredNodeMap)
                        not_special_list.append(0)
                    if ncost == -2:
                        # ncost == -2 means special node for loss3, i.e., cross entropy between predicted node mapping and ground-truth node mapping
                        # it does not matter we regard the special node as type 1 or type 0
                        # as there is no edge going out from a special node
                        ntypelist.append(0)
                        special_cost_list.append(0)
                        if oldID == '--1':
                            special_cost_list_for_loss3.append([1]*numPredNodeMap) 
                        else:
                            if oldID == '--2':
                                special_cost_list_for_loss3.append([0]*numPredNodeMap) 
                            else:
                                print("Error! oldID is not --1 or --2")
                                exit(-1)
                        not_special_list.append(0)
                    
                    cout_of_diff_type_neigh = [0, 0] #we just have two types: 0-type and 1-type
                    in_neighbors = g[oldID]
                    for in_neigh in in_neighbors:
                        in_neigh_ncost = g.nodes[in_neigh]['att']
                        if in_neigh_ncost == 0:
                            cout_of_diff_type_neigh[0] = cout_of_diff_type_neigh[0] + 1
                        if in_neigh_ncost == 1:
                            cout_of_diff_type_neigh[1] = cout_of_diff_type_neigh[1] + 1
                    neigh_type_count_list.append(cout_of_diff_type_neigh)
                    
                niddict = {"id" : torch.tensor(nidlist)}
                ntypedict = {"type" : torch.tensor(ntypelist)}
                ncostdict = {"cost" : torch.tensor(ncostlist)}
                special_cost_dict = {"special_cost" : torch.tensor(special_cost_list)}
                special_cost_for_loss3_dict = {"special_cost_for_loss3": torch.tensor(special_cost_list_for_loss3)}
                not_special_dict = {"non_special_node" : torch.tensor(not_special_list)}
                neigh_typle_count_dict = {"neigh_type_count" : torch.tensor(neigh_type_count_list)}
                dg.ndata.update(niddict)
                dg.ndata.update(ntypedict)
                dg.ndata.update(ncostdict)
                dg.ndata.update(special_cost_dict)
                dg.ndata.update(special_cost_for_loss3_dict)
                dg.ndata.update(not_special_dict)
                dg.ndata.update(neigh_typle_count_dict)
               

                # set edge type and cost
                etypelist = []
                ecostlist = []
                edge_norm_tmp = {} # for the end node of each edge, we count the number of incoming edges of different types
                for i in range(0, dg.number_of_edges()):
                    end1 = edges[0][i]
                    end2 = edges[1][i]
                    oldID_end1 = newID2oldID[end1]
                    oldID_end2 = newID2oldID[end2]
                    ecost = g[oldID_end1][oldID_end2].get("att")
                    ecostlist.append(ecost)
                    if ecost == 1 or ecost == 0:
                        etype = 0
                        etypelist.append(etype)

                        if end2 not in edge_norm_tmp:
                            edge_norm_tmp[end2] = {}
                            edge_norm_tmp[end2][etype] = 1
                        else:
                            if etype not in edge_norm_tmp[end2]:
                                edge_norm_tmp[end2][etype] = 1
                            else:
                                edge_norm_tmp[end2][etype] = edge_norm_tmp[end2][etype] + 1

                    if ecost == 10:
                        etype = 1
                        etypelist.append(etype)
                
                        if end2 not in edge_norm_tmp:
                            edge_norm_tmp[end2] = {}
                            edge_norm_tmp[end2][etype] = 1
                        else:
                            if etype not in edge_norm_tmp[end2]:
                                edge_norm_tmp[end2][etype] = 1
                            else:
                                edge_norm_tmp[end2][etype] = edge_norm_tmp[end2][etype] + 1


                # calculate edge norm
                etypenormlist = []
                for i in range(0, dg.number_of_edges()):
                    end1 = edges[0][i]
                    end2 = edges[1][i]
                    etype = etypelist[i]
                    edge_norm = edge_norm_tmp[end2][etype]
                    etypenormlist.append(1.0/edge_norm)


                # set edge weight for loss3
                edge_w_for_loss3 = []
                for i in range(0, dg.number_of_edges()):
                    end1 = edges[0][i]
                    end2 = edges[1][i]
                    oldID_end1 = newID2oldID[end1]
                    oldID_end2 = newID2oldID[end2]
                    if g.nodes[oldID_end2]['att'] == -2: 
                        # need to check if end1 is in ground-truth node mapping
                        if oldID_end2 == '--1' and oldID_end1 in nodeMap1_set:
                            edge_w_for_loss3.append(1)
                        else:
                            if oldID_end2 == '--2' and oldID_end1 in nodeMap2_set:
                                edge_w_for_loss3.append(1) 
                            else:
                                edge_w_for_loss3.append(0)
                    else:
                        edge_w_for_loss3.append(0)


                ecostdict = {'cost': torch.tensor(ecostlist)}
                etypedict = {'type': torch.tensor(etypelist)}
                etypenormdict = {'norm': torch.tensor(etypenormlist)}
                edge_w_for_loss3_dict = {'w4loss3': torch.tensor(edge_w_for_loss3)}
                
                dg.edata.update(ecostdict)
                dg.edata.update(etypedict) 
                dg.edata.update(etypenormdict)
                dg.edata.update(edge_w_for_loss3_dict)


                self.labels.append(float(g.graph.get("label")))
                dg2 = dg.to(torch.device('cuda:'+str(GPUID)))
                self.graphs.append(dg2)
                self.gids.append(g.graph.get("id"))
                print("dg nodes: ", dg.number_of_nodes(), " edges: ", dg.number_of_edges())



        self.labels = torch.tensor(self.labels).cuda()
        
    
        print(len(self.graphs))
        print(len(self.labels))
        print(len(self.gids))

        

    def save(self):
        pass

    def load(self):
        pass

    def has_cache(self):
        pass


def processNegOneLeftNodeInNodeMap(nodeMap):
    neg_in_nodeMap = []
    nodeMap_new = []
    for ele in nodeMap:
        abc = ele.split("|")
        if int(abc[0]) == -1:
            neg_in_nodeMap.append(int(abc[1]))
        else:
            nodeMap_new.append(ele)
    neg_in_nodeMap.sort()
    for i in range(0, len(neg_in_nodeMap)):
        nodeMap_new.append("-"+str(i+1)+"|"+str(neg_in_nodeMap[i]))
    
    return nodeMap_new
