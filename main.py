
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import dgl.function as fn
from functools import partial
from Dataset import GINDataset
import dgl
from torch.utils.data import DataLoader
import numpy as np
import time
import random

GPUID=2

class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_node_types, num_edge_types, num_bases=-1, bias=None,
                 activation=None, is_input_layer=False):
        super(RGCNLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_node_types = num_node_types
        self.num_rels = num_edge_types
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.is_input_layer = is_input_layer

        # sanity check
        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels

        self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat,
                                                self.out_feat)) # it is weight for edge type

        self.weight_for_node_type = nn.Parameter(torch.Tensor(self.num_node_types, self.in_feat, self.out_feat)) # 2 by d dimensional


        if self.num_bases < self.num_rels:
            self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))

        # add bias
        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(out_feat))

        # init trainable parameters
        nn.init.xavier_uniform_(self.weight_for_node_type,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.weight,
                                gain=nn.init.calculate_gain('relu'))
        if self.num_bases < self.num_rels:
            nn.init.xavier_uniform_(self.w_comp,
                                    gain=nn.init.calculate_gain('relu'))
        if self.bias:
            nn.init.xavier_uniform_(self.bias,
                                    gain=nn.init.calculate_gain('relu'))

    def forward(self, g):

        if self.num_bases < self.num_rels:
            weight = self.weight.view(self.in_feat, self.num_bases, self.out_feat)
            weight = torch.matmul(self.w_comp, weight).view(self.num_rels,
                                                        self.in_feat, self.out_feat)
        else:
            weight = self.weight

        if self.is_input_layer:
            def message_func(edges):
                # for the first layer, give a weight matrix to each node type
                w_for_node_type = self.weight_for_node_type[edges.src['type']]
                h_abc = torch.bmm(edges.src['h'].unsqueeze(1), w_for_node_type).squeeze()

                w = weight[edges.data['type']]
                msg = torch.bmm(h_abc.unsqueeze(1), w).squeeze()
                msg = msg * edges.data['norm'].unsqueeze(1)
                return {'msg': msg}

        else:
            def message_func(edges):
                w = weight[edges.data['type']]
                msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze()
                msg = msg.view(-1, self.out_feat) # need it. Otherwise, when self.out_feat=1, msg will be a 1D tensor
                msg = msg * edges.data['norm'].unsqueeze(1)
                return {'msg': msg}

        def apply_func(nodes):
            h = nodes.data['h']
            if self.bias:
                h = h + self.bias
            if self.activation:
                h = self.activation(h)
            return {'h': h}

        g.update_all(message_func, fn.sum(msg='msg', out='h'), apply_func)




class OutputLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_of_node_types, num_of_edge_types, num_bases=-1, bias=None,
                 activation=None, is_input_layer=False):
        super(OutputLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_of_node_types = num_of_node_types
        self.num_rels = num_of_edge_types
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.is_input_layer = is_input_layer

        self.count_reduced_nodes = 0

        # sanity check
        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels

        self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat,
                                                self.out_feat))

        if self.num_bases < self.num_rels:
            self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))

        # add bias
        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(out_feat))

        # init trainable parameters
        nn.init.xavier_uniform_(self.weight,
                                gain=nn.init.calculate_gain('relu'))
        if self.num_bases < self.num_rels:
            nn.init.xavier_uniform_(self.w_comp,
                                    gain=nn.init.calculate_gain('relu'))
        if self.bias:
            nn.init.xavier_uniform_(self.bias,
                                    gain=nn.init.calculate_gain('relu'))



    def forward(self, g):
        
        def message_func(edges):
            # process for loss2: sumOneConstraint
            abc = edges.src['pred_prob'].squeeze()
            return {'msg': abc}

        def reduce_func(nodes):
            return {'loss_for_sumOneConst': torch.sum(nodes.mailbox['msg'], dim=1)}

        def apply_func(nodes):
            h = nodes.data['loss_for_sumOneConst'] - 1.0
            h = torch.pow(h, 2)
            h = h * nodes.data['special_cost']
            return {'loss_for_sumOneCost2': h}


        g.update_all(message_func, reduce_func, apply_func)



###############################################################################
# Overall model defined as follows 


class Model(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, num_node_types, num_edge_types,
                 num_bases=-1, num_hidden_layers=1, num_predNodeMaps=1):
        super(Model, self).__init__()
        # self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_node_types = num_node_types
        self.num_rels = num_edge_types
        self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers
        self.num_predNodeMaps = num_predNodeMaps

        self.fc = nn.Linear(2, h_dim, bias=True)
        
        self.delete_gt_matching_rows = {} #key is gid, value is the list of row_ids

        self.build_model()


        self.fc2 = nn.Linear(h_dim, 1, bias=True)
        self.fc3 = nn.Linear(h_dim, 1, bias=True)

        self.fc_for_loss3 = nn.Linear(h_dim, num_predNodeMaps, bias=True).cuda() 
       
        self.outputLayer = OutputLayer(self.h_dim, self.h_dim, self.num_node_types, self.num_rels, self.num_bases,
                         activation=None)
 

    def build_model(self):
        self.layers = nn.ModuleList()
        first_hidden_layer = self.build_input_layer()
        self.layers.append(first_hidden_layer) 


    # initialize feature for each node
    def create_features(self):
        features = torch.arange(self.num_nodes)
        return features


    def build_input_layer(self):
        return RGCNLayer(self.h_dim, self.h_dim, self.num_node_types, self.num_rels, self.num_bases,
                         activation=F.relu, is_input_layer=True)

    def build_hidden_layer(self):
        return RGCNLayer(self.h_dim, self.h_dim, self.num_node_types, self.num_rels, self.num_bases,
                         activation=F.relu)

    def build_output_layer(self):
        return RGCNLayer(self.h_dim, self.h_dim, self.num_node_types, self.num_rels, self.num_bases,
                         activation=F.relu)

    def forward(self, g, glabel, groundTruthNodeMaps, gids):      

        # h = g.in_degrees().view(-1, 1).float()
        h = g.ndata["neigh_type_count"].float().cuda() # using "neigh_type_count" is better than just using "node_degree"

        g.ndata['h'] = self.fc(h)

        for layer in self.layers:
            layer(g)
        
        fc_res = self.fc2(g.ndata['h'])

        g.ndata['pred_prob'] = torch.clamp(torch.sigmoid(fc_res), 1e-7, 1e10) 
        
        

        loss6_out_no_grad_fn = []
        # productive loss2 of GLMNet: Graph Learning-Matching Networks for Feature Matching
        # and additive loss2
        # additive loss2 is L_qip in paper; productive loss2 is L_cr in paper
        productive_loss2_has_no_grad_fn = []
        sumOneLoss_no_grad_fun = [] 
        glist = dgl.unbatch(g)
        for gidx in range(0, len(glist)):
            cur_g = glist[gidx]
         
            allGTNodeMaps_for_curG = [groundTruthNodeMaps[gidx]]
            matrix_size = allGTNodeMaps_for_curG[0].shape[0] # nodeMap is a matrix_size by matrix_size map
            pred_nodeMaps = cur_g.ndata['pred_prob'][0:matrix_size*matrix_size]

            # we predict several nodeMaps for a graph
            # each predicted nodeMap has a crossEntropy with the groundTruthNodeMap
            ceList_of_curG = []
            for curTrueNodeMap in allGTNodeMaps_for_curG:
                pred_nodeMaps = pred_nodeMaps.view(-1, matrix_size)
                pred_nodeMaps_sinkhorn = pred_nodeMaps
   
                ce = curTrueNodeMap * torch.log2(pred_nodeMaps_sinkhorn) 
                ce = -1.0*ce
                ce = torch.sum(ce)
                ceList_of_curG.append(ce)

                constraintRegularizedLoss_of_curG = get_constrain_regularizer_loss(pred_nodeMaps_sinkhorn)
                productive_loss2_has_no_grad_fn.append(constraintRegularizedLoss_of_curG)

                sumOneLoss_curG = getSumOne_loss(pred_nodeMaps_sinkhorn)
                sumOneLoss_no_grad_fun.append(sumOneLoss_curG)

            minCE = ceList_of_curG[0]
            loss6_out_no_grad_fn.append(minCE)


        loss6_out_has_grad_fn = loss6_out_no_grad_fn[0].view(1,)
        for idx in range(1, len(loss6_out_no_grad_fn)):
            loss6_out_has_grad_fn = torch.cat([loss6_out_has_grad_fn, loss6_out_no_grad_fn[idx].view(1,)])

        productive_loss2_has_grad_fn = productive_loss2_has_no_grad_fn[0].view(1,)
        for idx in range(1, len(productive_loss2_has_no_grad_fn)):
            productive_loss2_has_grad_fn = torch.cat([productive_loss2_has_grad_fn, productive_loss2_has_no_grad_fn[idx].view(1,)])
        
        sumOneLoss_has_grad_fun = sumOneLoss_no_grad_fun[0].view(1,)
        for idx in range(1, len(sumOneLoss_no_grad_fun)):
            sumOneLoss_has_grad_fun = torch.cat([sumOneLoss_has_grad_fun, sumOneLoss_no_grad_fun[idx].view(1,)])
        
        fc_res_for_loss3 = self.fc_for_loss3(g.ndata['h'])
        prob_for_loss3 = torch.sigmoid(fc_res_for_loss3)

        g.ndata['prob_for_loss3'] = prob_for_loss3.cuda()
   
        self.outputLayer(g)
    
        gembed = dgl.mean_nodes(g, 'h')
        pred_obj_of_qp = self.fc3(gembed)
        pred_obj_of_qp = pred_obj_of_qp.squeeze()

        res = [pred_obj_of_qp, sumOneLoss_has_grad_fun, productive_loss2_has_grad_fn, loss6_out_has_grad_fn]

        return res
        

###############################################################################

def getSumOne_loss(x):
    row_sum = torch.sum(x,1)
    col_sum = torch.sum(x,0)

    ones = torch.ones(x.shape[0]).cuda()

    row_sum_diff = row_sum - ones
    col_sum_diff = col_sum - ones

    row_sum_diff_square = torch.pow(row_sum_diff, 2)
    col_sum_diff_square = torch.pow(col_sum_diff, 2)

    a = torch.sum(row_sum_diff_square)
    b = torch.sum(col_sum_diff_square)
    
    return a+b



def get_constrain_regularizer_loss(x):
    row_sum = torch.sum(x,1)
    row_sum = row_sum.view(-1,1)
    col_sum = torch.sum(x,0)
    col_sum = col_sum.view(1,-1)
    x2 = x - row_sum
    x3 = x - col_sum
    x2 = -x2
    x3 = -x3
    y = x * x2
    y2 = x * x3
    z = torch.sum(y)
    z2 = torch.sum(y2)

    return z+z2
    


cudaID = "cuda:"+str(GPUID)

def collate(samples):
    graphs, labels, groundTruthNodeMaps, allGTNodeMaps, gids = map(list, zip(*samples))

    graphs2 = []
    for g in graphs:
        g2 = g.to(torch.device(cudaID))
        graphs2.append(g2)
    
    batched_graph = dgl.batch(graphs2)
    batched_labels = torch.tensor(labels).cuda()
    


    return batched_graph, batched_labels, groundTruthNodeMaps, allGTNodeMaps, gids


def my_loss(x, y, hasLoss3, hasLoss2, hasLoss2_constraintRegularizer):
    # x is prediction, y is target
    loss_for_qp_obj = torch.pow((x[0]-y), 2) # L_d in paper
    loss_for_sum_one_const = x[1] # L_qip in paper
    constraintRegularizer = x[2] # L_cr in paper
    loss6 = x[3] # L_m in paper


    if hasLoss3 and hasLoss2 and hasLoss2_constraintRegularizer:
        print("loss1 + loss2 + loss2v2 + loss3")
        loss = loss_for_qp_obj + loss_for_sum_one_const + constraintRegularizer + loss6
    
        print("loss1", loss_for_qp_obj)
        print("loss2", loss_for_sum_one_const)
        print("loss2v2", constraintRegularizer)
        print('loss6', loss6)
    
        loss = loss.float()

    elif hasLoss3 == False and hasLoss2 == False and hasLoss2_constraintRegularizer:
        print("loss1 + loss2v2")
        print("loss1", loss_for_qp_obj)
        print("loss2v2", constraintRegularizer)
        loss = loss_for_qp_obj + constraintRegularizer
        loss = loss.float()

    elif hasLoss3 and hasLoss2 == False and hasLoss2_constraintRegularizer == False:
        print("loss1 + loss3")
        print("loss1", loss_for_qp_obj)
        print('loss6', loss6)
        loss = loss_for_qp_obj + loss6
        loss = loss.float()

    elif hasLoss3 == False and hasLoss2 == True and hasLoss2_constraintRegularizer == False:
        print("loss1 + loss2")
        print("loss1", loss_for_qp_obj)
        print('loss2', loss_for_sum_one_const)
        loss = loss_for_qp_obj + loss_for_sum_one_const
        loss = loss.float()

    elif hasLoss3 == False and hasLoss2 == True and hasLoss2_constraintRegularizer == True:
        print("loss1 + loss2v1v2")
        print("loss1", loss_for_qp_obj)
        print('loss2', loss_for_sum_one_const)
        print("loss2v2", hasLoss2_constraintRegularizer)
        loss = loss_for_qp_obj + loss_for_sum_one_const + constraintRegularizer
        loss = loss.float()

    elif hasLoss3 == False and hasLoss2 == False and hasLoss2_constraintRegularizer == False:
        print("just loss1")
        loss = loss_for_qp_obj 
        loss = loss.float()

    elif hasLoss3 == True and hasLoss2 == False and hasLoss2_constraintRegularizer == True:
        print("loss1 + loss2v2 + loss3")
        loss = loss_for_qp_obj + constraintRegularizer + loss6
        loss = loss.float()

    elif hasLoss3 == True and hasLoss2 == True and hasLoss2_constraintRegularizer == False:
        loss = loss_for_qp_obj + loss_for_sum_one_const + loss6
        print("loss1", loss_for_qp_obj)
        print("loss2v1", loss_for_sum_one_const)
        print('loss3', loss6)
        loss = loss.float()

    else:
        print("loss error!")
        exit(-1)

    return torch.mean(loss)


def my_loss_for_test(x, y):
    # x is prediction, y is target
    loss_for_qp_obj = torch.pow((x[0]-y), 2) 
    mm = x[0]-y

    return [torch.mean(loss_for_qp_obj), torch.mean(mm), torch.std(mm)]




###############################################################################
# Create graph and model
# ~~~~~~~~~~~~~~~~~~~~~~~

# configurations
n_hidden = 32 # number of hidden units
out_dim = 1 # regression problem
n_bases = -1 # use number of relations as number of bases
n_hidden_layers = 0 # use 1 input layer, 1 output layer, no hidden layer
n_epochs = 8000 # epochs to train, can be early stopped
lr = 0.005 # learning rate
l2norm = 0 # L2 norm coefficient


num_predNodeMaps = 1000 # predict how many nodeMaps. it is not really used in code. 
                        # but, because of historical reason, it should be set at least=1, otherwise, the code cannot run
                        # if want 0, please set "hasLoss3 = False"
hasLoss2 = False # L_qip in paper
loss2_productive = True # L_cr in paper
hasLoss3 = False # L_m in paper

has_sinkhorn = False 
sinkhornTemperature = 0.1
delete_gt_matching_percent = 0 # randomly delete some elements in the ground-truth node matching

refine = False
validNonSeenGraphs = False
testNonSeenGraphs = False


torch.cuda.set_device(GPUID)


seed = int(num_predNodeMaps * lr*1000 * sinkhornTemperature*100)
print('seed ', seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)            
torch.cuda.manual_seed(seed)       
torch.cuda.manual_seed_all(seed)   



dataSetName = "IMDB" #"PTC" #"Linux" # "IMDB" # AIDS
dirName = "asso_graph_example/IMDB.s30" # "PTC" #"linuxnoL.s30.sample" #"IMDB.s30" # aidsG.noL.s30

train_num = 100
valid_num = 50
test_num = 50

print("load graph for training...")
dataset = GINDataset(dataSetName, self_loop=False, raw_dir=dirName+".prodG.2000.txt", start=0, end=train_num, num_predNodeMap=num_predNodeMaps)

print('load graph for valid...')
valid_dataset = GINDataset(dataSetName, self_loop=False, raw_dir=dirName+".prodG.2000.txt", start=train_num, end=train_num+valid_num, num_predNodeMap=num_predNodeMaps)


print('load graph for test...')
test_dataset = GINDataset(dataSetName, self_loop=False, raw_dir=dirName+".prodG.2000.txt", start=train_num+valid_num, end=train_num+valid_num+test_num, num_predNodeMap=num_predNodeMaps)



dataloader = DataLoader(
    dataset,
    batch_size=1,
    collate_fn=collate,
    drop_last=False,
    shuffle=False)



validdataloader = DataLoader(
    valid_dataset,
    batch_size=1,
    collate_fn=collate,
    drop_last=False,
    shuffle=False)

testdataloader = DataLoader(
    test_dataset,
    batch_size=1,
    collate_fn=collate,
    drop_last=False,
    shuffle=False)



num_rels = 2
num_node_types = 2



# create model
model = Model(n_hidden,
              n_hidden,
              out_dim,
              num_node_types,
              num_rels,
              num_bases=n_bases,
              num_hidden_layers=n_hidden_layers,
              num_predNodeMaps=num_predNodeMaps)

model.cuda()

print(model)


def adjust_learning_rate(optimizer, epoch, avg_mse_of_test):
    """Sets the learning rate to the initial LR decayed"""
    old_lr = optimizer.param_groups[0]['lr']
    if epoch < 50:
        pass
    else:
        if epoch < 1000:
            if epoch % 5 == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = old_lr * 0.96

    

###############################################################################
# Training loop
# ~~~~~~~~~~~~~~~~

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2norm)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma = 0.9)

if hasLoss2 and hasLoss3 and loss2_productive:
    loss_str = "loss1_2v1v2_3"
if hasLoss2 == False and loss2_productive and hasLoss3 == False :
    loss_str = "loss1_2v2"
if hasLoss2 == False and loss2_productive == False and hasLoss3:
    loss_str = "loss1_3"
if hasLoss2 == True and loss2_productive == False and hasLoss3 == False:
    loss_str = "loss1_2"
if hasLoss2 == True and loss2_productive == True and hasLoss3 == False:
    loss_str = "loss1_2v1v2"
if hasLoss2 == False and loss2_productive == False and hasLoss3 == False:
    loss_str = "loss1"
if hasLoss2 == False and loss2_productive == True and hasLoss3 == True:
    loss_str = "loss1_2v2_3"
if hasLoss2 == True and loss2_productive == False and hasLoss3 == True:
    loss_str = "loss1_2v1_3"

f = open("res/"+dataSetName+"_h1_"+loss_str+"_nodeMap"+str(num_predNodeMaps)+"_lr"+str(lr)+"_train"+str(train_num)+"val"+str(valid_num)+"test"+str(test_num)+".txt", "w")
f.write("num_predNodeMaps "+str(num_predNodeMaps)+" lr "+str(lr)+" seed "+str(seed)+"\n")

print("start training...")
train_time_cost = 0
test_time_cost = 0

MSE_validData_epoch_list = []

with torch.autograd.set_detect_anomaly(True):   
    epoch_id = 0
    batch_id = 0
    for epoch in range(n_epochs):
        print("dataSet: ", dataSetName)
        print("hasLoss2", hasLoss2)
        print("loss2_productive", loss2_productive)
        print("hasLoss3", hasLoss3)
        print("has_sinkhorn", has_sinkhorn)
        print("delete_gt_matching_percent", delete_gt_matching_percent)

        model.train()
        batch_count = 0
        avg_loss_of_train = 0
        start_train_time = time.time()
        for batched_graph, labels, groundTruthNodeMaps, allGTNodeMaps, graphIDs in dataloader:
            print('='*40+" epoch ", epoch, "batch ", batch_count, "nM ", num_predNodeMaps, "lr ",lr, "t ", sinkhornTemperature, "seed ", seed)
            # print("batched_graph : ")
            # print(batched_graph)
            print("labels : ")
            print(labels)
            pred = model(batched_graph, labels, groundTruthNodeMaps, graphIDs)
            print('pred')
            print(pred)
            
            loss = my_loss(pred, labels, hasLoss3, hasLoss2, loss2_productive)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()       

            print("train loss ", loss) 
            avg_loss_of_train = avg_loss_of_train + loss.item()      
            batch_count = batch_count + 1
        avg_loss_of_train = avg_loss_of_train / batch_count
        end_train_time = time.time()
        train_time_cost = train_time_cost + (end_train_time - start_train_time)


        # do a test after an epoch
        print("do test ....")
        model.eval()

        valid_count = 0
        avg_mse_of_valid = 0


        test_count = 0
        avg_mse_of_test = 0
        avg_raw_diff_of_test = 0
        avg_std_of_raw_diff_of_test = 0
        avg_ce_of_test = 0
        
        with torch.no_grad():
            # do valid
            print("start valid ................")
            for valid_graph, valid_label, groundTruthNodeMaps, allGTNodeMaps, graphIDs in validdataloader:             
                valid_pred = model(valid_graph, valid_label, groundTruthNodeMaps, graphIDs)
                tmp = my_loss_for_test(valid_pred, valid_label)
                valid_loss = tmp[0]
                print('valid loss ', valid_loss)
                avg_mse_of_valid = avg_mse_of_valid + valid_loss.item() 
                valid_count = valid_count + 1
                print("+++++++"*5, "valid finish")



            # do test
            print("start test ................")
            for test_graph, test_label, groundTruthNodeMaps, allGTNodeMaps, graphIDs in testdataloader:
                # print(test_graph)
                # print(test_label)
                
                start_test_time = time.time()
                test_pred = model(test_graph, test_label, groundTruthNodeMaps, graphIDs)
                end_test_time = time.time()
                test_time_cost = test_time_cost + (end_test_time - start_test_time)

                # print('test_pred')
                # print(test_pred)
                
                tmp = my_loss_for_test(test_pred, test_label)
                test_loss = tmp[0]
                mean_raw_diff_of_MSE = tmp[1].item() # raw_diff_of_mse means: pred_mse - real_mse
                std_raw_diff_of_MSE = tmp[2].item()
                print('test loss ', test_loss)
                print('mean_raw_diff_of_MSE ', mean_raw_diff_of_MSE)
                print('std_raw_diff_of_MSE ', std_raw_diff_of_MSE)
                avg_mse_of_test = avg_mse_of_test + test_loss.item()
                avg_raw_diff_of_test = avg_raw_diff_of_test + mean_raw_diff_of_MSE
                avg_std_of_raw_diff_of_test = avg_std_of_raw_diff_of_test + std_raw_diff_of_MSE
                test_count = test_count + 1
                print("+++++++"*5, 'test finish')

        avg_mse_of_test = avg_mse_of_test / test_count
        avg_raw_diff_of_test = avg_raw_diff_of_test / test_count
        avg_std_of_raw_diff_of_test = avg_std_of_raw_diff_of_test / test_count
        avg_ce_of_test = avg_ce_of_test / test_count

        avg_mse_of_valid = avg_mse_of_valid / valid_count

        MSE_validData_epoch_list.append(avg_mse_of_valid)
        


        f.write(str(epoch)+" "+str(avg_loss_of_train)+" "+str(avg_mse_of_valid)+" "+str(avg_mse_of_test)+"\n")
        f.flush()

        if epoch > 600:
            # if MSE is not better in 10 epochs, early stop
            aaa = np.array(MSE_validData_epoch_list[-10:])
            bbb = np.argmin(aaa)
            ccc = np.min(aaa)
            if ccc < avg_mse_of_valid and bbb == 0:
                print("early stop at epoch: ", epoch)
                break

        epoch_id = epoch_id + 1

        adjust_learning_rate(optimizer, epoch, avg_mse_of_test)
        cur_lr = optimizer.param_groups[0]['lr']
        print("cur lr: ", cur_lr)
    

    print("train time (s) ", train_time_cost)
    print("test time (s) ", test_time_cost)

    f.write("train time(s) " + str(train_time_cost)+"\n")
    f.write("test time(s) " + str(test_time_cost)+"\n")

    f.close()

