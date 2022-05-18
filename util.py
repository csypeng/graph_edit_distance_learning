import networkx as nx

# In this code of computing product graph, g1 and g2 are assumed having no node label
def getProdGraph(g1, g2):
    small_g = None
    large_g = None
    if g1.number_of_nodes() <= g2.number_of_nodes():
        small_g = nx.Graph(g1)
        large_g = nx.Graph(g2)
    else:
        small_g = nx.Graph(g2)
        large_g = nx.Graph(g1)

    originalSmallGSize = small_g.number_of_nodes()
    originalLargeGSize = large_g.number_of_nodes()

    # add dummy nodes
    dummy_count = large_g.number_of_nodes() - small_g.number_of_nodes()
    while dummy_count > 0:
        small_g.add_node(str(-dummy_count), attr='-1')
        dummy_count = dummy_count - 1

    # prod graph
    m1 = {} # small_g node -> prod_g node
    m2 = {} # large_g node -> prod_g node
    prod_g = nx.Graph(id = small_g.graph.get("id")+"|"+large_g.graph.get("id"), size = str(originalSmallGSize)+" "+str(originalLargeGSize))
    print(small_g.number_of_nodes())
    print(small_g.number_of_edges())
    print('----')
    print(large_g.number_of_nodes())
    print(large_g.number_of_edges())
    print("++++")
    for n1 in small_g.nodes():
        for n2 in large_g.nodes():
            prod_node = n1+"|"+n2
            ncost = 0
            if int(n1)<0 and int(n2)>=0:
                ncost = 1
            if int(n1)>=0 and int(n2)<0:
                ncost = 1
            prod_g.add_node(prod_node, w=ncost)
            
            if n1 not in m1:
                x = [prod_node]
                m1[n1] = x
            else:
                m1[n1].append(prod_node)

            if n2 not in m2:
                x = [prod_node]
                m2[n2] = x
            else:
                m2[n2].append(prod_node)


    for e1 in small_g.edges():
        end1 = e1[0]
        end2 = e1[1]
        for a in large_g.nodes():
            for b in large_g.nodes():
                if a != b:
                    if large_g.has_edge(a,b) == False:
                        prod_g.add_edge(end1+"|"+a, end2+"|"+b, w=1)

    for e2 in large_g.edges():
        end1 = e2[0]
        end2 = e2[1]
        for a in small_g.nodes():
            for b in small_g.nodes():
                if a != b:
                    if small_g.has_edge(a,b) == False:
                        prod_g.add_edge(a+"|"+end1, b+"|"+end2, w=1)


   

    # add special nodes for the SumEqualsToOne constraint
    count_of_special_nodes = 1
    for x in m1.values():
        # print(x)
        cur_special_node = str(-count_of_special_nodes)
        # print('cur_special_node', cur_special_node)
        prod_g.add_node(cur_special_node, w = -1) # weight = -1 means special node
        for node in x:
            prod_g.add_edge(node, cur_special_node, w = 0) # weight = 0 here is just a place holder
        count_of_special_nodes = count_of_special_nodes + 1
    for x in m2.values():
        # print(x)
        cur_special_node = str(-count_of_special_nodes)
        prod_g.add_node(cur_special_node, w = -1) # weight = -1 means special node
        for node in x:
            prod_g.add_edge(node, cur_special_node, w = 0) # weight = 0 here is just a place holder
        count_of_special_nodes = count_of_special_nodes + 1

    print("prod_g nodes: ", prod_g.number_of_nodes())
    print("prod_g edges: ", prod_g.number_of_edges())
    
    return prod_g
