import numpy as np
import copy
import os
import networkx as nx
from networkx.algorithms import isomorphism
from networkx.generators import classic
from collections import defaultdict
import utils

base = "../local_explanations/"

house = nx.Graph()
house.add_edge(0,1)
house.add_edge(0,2)
house.add_edge(2,1)
house.add_edge(2,3)
house.add_edge(1,4)
house.add_edge(4,3)
grid  = nx.grid_2d_graph(3, 3)
wheel = classic.wheel_graph(6)


pattern_no2 = nx.Graph()
pattern_no2.add_nodes_from([
    (0, {"atom_type": 1}),
    (1, {"atom_type": 4}),
    (2, {"atom_type": 1})
])
pattern_no2.add_edges_from([(0,1), (1,2)])

pattern_nh2 = nx.Graph()
pattern_nh2.add_nodes_from([
    (0, {"atom_type": 3}),
    (1, {"atom_type": 4}),
    (2, {"atom_type": 3})
])
pattern_nh2.add_edges_from([(0,1), (1,2)])

bamultishapes_classes_names = ["house", "grid", "wheel", "ba", "house+grid", "house+wheel", "wheel+grid", "all"]
mutag_classes_names = ["NO2", "Others"]
priori_etn_classes_names = ["M", "N", "NA", "PAT", "NP", "MN", "MP", "OTHERS"]
posteriori_etn_classes_names = ["N", "M", "NA", "NP", "NM", "MA", "NMA", "O"]


def elbow_method(weights, index_stopped=None, min_num_include=7, backup=None):
    sorted_weights = sorted(weights, reverse=True)
    sorted_weights = np.convolve(sorted_weights, np.ones(min_num_include), 'valid') / min_num_include

    stop = np.mean(sorted_weights) if backup is None else backup # backup threshold
    for i in range(len(sorted_weights)-2):
        if i < min_num_include:
            continue
        if sorted_weights[i-1] - sorted_weights[i] > 0.0:
            if sorted_weights[i-1] - sorted_weights[i] >= 40 * (sorted_weights[0] - sorted_weights[i-2]) / 100 + (sorted_weights[0] - sorted_weights[i-2]):
                stop = sorted_weights[i]
                if index_stopped is not None:
                    index_stopped.append(stop)
                break
    return stop


def assign_class(pattern_matched):
    if len(pattern_matched) == 0: #ba
        return 3
    elif len(pattern_matched) == 1: #single motif
        return pattern_matched[0]
    else:
        assert len(pattern_matched) <= 3
        if 0 in pattern_matched and 1 in pattern_matched and 2 in pattern_matched:
            return 7
        elif 0 in pattern_matched and 1 in pattern_matched: 
            return 4
        elif 0 in pattern_matched and 2 in pattern_matched: 
            return 5
        elif 1 in pattern_matched and 2 in pattern_matched: 
            return 6


def label_explanation(G, house, grid, wheel, return_raw=False):
    pattern_matched = []
    for i , pattern in enumerate([house, grid, wheel]):
        GM = isomorphism.GraphMatcher(G, pattern)
        if GM.subgraph_is_isomorphic():
            pattern_matched.append(i)
    if return_raw:
        return pattern_matched
    else:
        return assign_class(pattern_matched)


def label_explanations(adjs, num_graphs):
    classes = []
    classes_names = ["house", "grid", "wheel", "ba", "house+grid", "house+wheel", "wheel+grid", "all"]
    for k in range(num_graphs):
        G = nx.from_numpy_matrix(adjs[k])
        c = label_explanation(G, house, grid, wheel)
        classes.append(c)
    classes = np.array(classes)
    return classes , classes_names

def evaluate_cutting(ori_adjs, adjs):
    num_shapes = 0
    for i , adj in enumerate(ori_adjs):    
        G = nx.from_numpy_matrix(adj)
        
        # count original patterns
        for pattern in [house, grid, wheel]:
            GM = isomorphism.GraphMatcher(G, pattern)
            match = list(GM.subgraph_isomorphisms_iter())
            if len(match) > 0:
                num_shapes += 1

    num_preserved = 0
    num_multipleshapes = 0
    for i , adj in enumerate(adjs):                
        G = nx.from_numpy_matrix(adj)
        for cc in nx.connected_components(G):
            if len(cc) > 2:
                G1 = G.subgraph(cc)
                pattern_found = False
                for pattern in [house, grid, wheel]:
                    GM = isomorphism.GraphMatcher(G1, pattern)
                    match = list(GM.subgraph_isomorphisms_iter())
                    if len(match) > 0:
                        if pattern_found:
                            num_multipleshapes += 1
                        num_preserved += 1
                        pattern_found = True                    
    print(f"Num shapes: {num_shapes}, Num Preserved: {num_preserved}, Ratio: {round(num_preserved/num_shapes, 3)}, Num Multipleshapes: {num_multipleshapes}")
    return round(num_preserved/num_shapes, 3)
    
##
# read local explanations and cut them based on a certain policy
##
def read_bamultishapes(explainer="PGExplainer", dataset="BAMultiShapes", model="GCN", split="TRAIN", evaluate_method=False, remove_mix=False, min_num_include=5, manual_cut=None):
    base_path = base + f"{explainer}/{dataset}/{model}/"
    adjs , edge_weights , index_stopped = [] , [] , []
    ori_adjs, ori_edge_weights, ori_classes , belonging , ori_predictions = [], [], [] , [] , []
    precomputed_embeddings , gnn_embeddings = [] , []
    total_graph_labels , total_cc_labels , le_classes = [] , [] , []
    
    global summary_predictions
    summary_predictions = {"correct": [], "wrong": []}

    num_multi_shapes_removed , num_class_relationship_broken , cont_num_iter , num_iter = 0 , 0 , 0 , 0
    for split in [split]:
        path = base_path + split + "/"
        #path_emb = base_emb + split + "/"
        for c in ["1","0"]:
            for pp in os.listdir(path + c + "/"):
                graph_id = int(pp.split(".")[0])
                adj = np.load(path + c + "/" + pp, allow_pickle=True)
                g = nx.from_numpy_array(adj)                 
                
                # ori_predictions.append(prediction)                
                # emb = np.load(path_emb + c + "/" + str(graph_id) + ".pkl", allow_pickle=True)
                # gnn_embeddings.append(emb)
                
                cut = elbow_method(np.triu(adj).flatten(), index_stopped, min_num_include) if manual_cut is None else manual_cut
                masked = copy.deepcopy(adj)
                masked[masked <= cut] = 0
                masked[masked > cut] = 1   
                G = nx.from_numpy_matrix(masked)

                added = 0
                graph_labels = label_explanation(g, house, grid, wheel, return_raw=True)
                gnn_pred = int(pp.split("_")[0])
                #if gnn_pred != int(c):
                #    summary_predictions["wrong"].append(assign_class(graph_labels))
                #    continue
                summary_predictions["correct"].append(assign_class(graph_labels))
                total_cc_labels.append([])
                cc_labels = []
                for cc in nx.connected_components(G):
                    if len(cc) > 2:
                        G1 = G.subgraph(cc)
                        if not nx.diameter(G1) == len(G1.edges()): #if is not a line  
                            cc_lbl = label_explanation(G1, house, grid, wheel, return_raw=True)
                            if remove_mix and assign_class(cc_lbl) >= 4:
                                num_multi_shapes_removed += 1
                                if added:
                                    print("added = ", added)
                                    del adjs[-1], edge_weights[-1], belonging[-1], total_cc_labels[-1], le_classes[-1]
                                    added = 0
                                break
                            added += 1
                            cc_labels.extend(cc_lbl)
                            total_cc_labels[-1].extend(cc_lbl) 
                            adjs.append(nx.to_numpy_matrix(G1))
                            edge_weights.append(nx.get_edge_attributes(G1,"weight"))    
                            belonging.append(num_iter)
                            le_classes.append(assign_class(cc_lbl))
                            
                            if gnn_embeddings != []:
                                nodes_to_keep = list(G1.nodes())
                                to_keep = gnn_embeddings[-1][nodes_to_keep]
                                precomputed_embeddings.append(to_keep)
                if total_cc_labels[-1] == []:
                    del total_cc_labels[-1]
                if added:
                    if graph_labels != []: total_graph_labels.append(graph_labels)
                    num_iter += 1
                    ori_adjs.append(adj)            
                    ori_edge_weights.append(nx.get_edge_attributes(g,"weight"))
                    ori_classes.append(gnn_pred) #c | gnn_pred
                    for lbl in graph_labels:
                        if lbl not in cc_labels:
                            num_class_relationship_broken += 1
                            break    
                cont_num_iter += 1       
    belonging = utils.normalize_belonging(belonging)
    if evaluate_method:
        evaluate_cutting(ori_adjs, adjs)
        print("num_class_relationship_broken: ", num_class_relationship_broken, " num_multi_shapes_removed:" , num_multi_shapes_removed)
    return adjs , edge_weights , ori_classes , belonging , summary_predictions, le_classes #(total_graph_labels, total_cc_labels)



##
# read local explanations and cut them based on a certain policy
##
def read_mutagenicity(explainer="PGExplainer", model="GCN_TF", split="TRAIN", evaluate_method=True, min_num_include=None, manual_cut=None):
    base_path = base + f"{explainer}/Mutagenicity/{model}/"
    adjs , edge_weights , index_stopped = [] , [] , []
    ori_adjs, ori_edge_weights, ori_classes , belonging  = [], [], [] , [] 
    precomputed_embeddings, ori_embeddings = [], []
    le_classes = []
    ori_idxs , nodes_kept = [] , []
    summary_predictions = defaultdict(list)

    cont_num_iter , num_iter = 0 , 0 
    for split in [split]:
        path = base_path + split + "/"
        for c in ["0", "1"]:
            for pp in os.listdir(path + c + "/"):
                graph_id = int(pp.split("_")[1].split(".")[0])
                gnn_pred = int(pp.split("_")[0])
                summary_predictions["correct"].append(int(c))
                adj = np.load(path + c + "/" + pp, allow_pickle=True)
                features = np.load(path + "features" + "/" + pp, allow_pickle=True)
                                
                if manual_cut is None:
                    cut = elbow_method(np.triu(adj).flatten(), index_stopped, min_num_include=2)
                else:
                    cut = manual_cut
                
                masked = copy.deepcopy(adj)
                masked[masked < cut] = 0
                masked[masked >= cut] = 1   
                G = nx.from_numpy_matrix(masked)

                connected_components = list(nx.connected_components(G))
                if len(connected_components) == adj.shape[0]: #only single nodes as connected component. No edges in the graph
                    continue
                else:
                    added = 0
                    for cc in connected_components:
                        if len(cc) >= 2: #to exclude single nodes
                            G1 = G.subgraph(cc)
                            if len(G1.nodes()) >= len(G.nodes()) - 3:
                                continue

                            nodes_to_keep = list(G1.nodes())
                            #to_keep = embedding[nodes_to_keep]
                            to_keep = features[nodes_to_keep]
                            for n in nodes_to_keep:
                                G1.nodes[n]["atom_type"] = features[n].argmax(-1)

                            pattern_found = False
                            for i , pattern in enumerate([pattern_nh2, pattern_no2]):
                                if pattern_found:
                                    continue
                                GM = isomorphism.GraphMatcher(G1, 
                                                            pattern,
                                                            node_match=isomorphism.categorical_node_match(['atom_type'], [0]))
                                if GM.subgraph_is_isomorphic():
                                    pattern_found = True
                                    le_classes.append(i)
                            if not pattern_found:
                                le_classes.append(2) 
                            if le_classes[-1] == 0: #temprary remove class 0 since there is just 1 example
                                del le_classes[-1]
                                continue

                            added += 1
                            adjs.append(nx.to_numpy_matrix(G1))
                            edge_weights.append(nx.get_edge_attributes(G1,"weight"))    
                            belonging.append(num_iter)
                            nodes_kept.append(nodes_to_keep)
                            ori_idxs.append(graph_id)
                            precomputed_embeddings.append(to_keep)                               
                                                      
                if added:
                    g = nx.from_numpy_array(adj)
                    num_iter += 1
                    ori_adjs.append(adj)
                    ori_embeddings.append(features)
                    ori_edge_weights.append(nx.get_edge_attributes(g,"weight"))
                    ori_classes.append(gnn_pred)
                else:
                    pass
                cont_num_iter += 1       
    belonging = utils.normalize_belonging(belonging)
    le_classes = [l-1 for l in le_classes]
    return adjs , edge_weights , ori_adjs , ori_classes , belonging , summary_predictions , le_classes, precomputed_embeddings


def convert_labels_to_names(attrs):
    ret = []
    for x in attrs:
        if x==0:
            ret.append("M")
        if x==1:
            ret.append("P")
        if x==2:
            ret.append("A")
        if x==3:
            ret.append("N")
        if x==4:
            ret.append("EGO")
    return ret


##
# read local explanations and cut them based on a certain policy
##
def read_hin(explainer="PGExplainer", model="GCN", split="TRAIN", min_num_include=2, manual_cut=None, priori_annotation=True):
    base_path = base + f"{explainer}/HIN/{model}/" #ETN_no_split_t1t2_val_643
    adjs , edge_weights , index_stopped = [] , [] , []
    ori_adjs, ori_edge_weights, ori_classes , belonging  = [], [], [] , [] 
    precomputed_embeddings, ori_embeddings = [], []
    le_classes = []
    ori_idxs , nodes_kept = [] , []
    summary_predictions = defaultdict(list)
    all_names = {}
    others = defaultdict(int)

    cont_num_iter , num_iter = 0 , 0 
    for split in [split]:
        path = base_path + split + "/"
        for c in ["0", "1"]:
            for pp in os.listdir(path + c + "/"):
                graph_id = int(pp.split("_")[1].split(".")[0])
                gnn_pred = int(pp.split("_")[0])
                #if gnn_pred != int(c):
                #    summary_predictions["wrong"].append(int(c))
                #    continue
                summary_predictions["correct"].append(int(c))
                adj = np.load(path + c + "/" + pp, allow_pickle=True)
                features = np.load(path + "features" + "/" + pp, allow_pickle=True)
                                
                if manual_cut is None:
                    cut = elbow_method(np.triu(adj).flatten(), index_stopped, min_num_include=min_num_include)
                else:
                    cut = manual_cut
                
                masked = copy.deepcopy(adj)
                masked[masked < cut] = 0
                masked[masked >= cut] = 1   
                G = nx.from_numpy_matrix(masked)

                ego_node = int(np.argmax(features[:, 4] == 1))
                assert sum(features[:, 4] == 1) == 1
                connected_components = list(nx.connected_components(G))
                added = 0
                for cc in connected_components:                    
                    if len(cc) >= 2 and ego_node in cc: #to exclude single nodes
                        G1 = G.subgraph(cc)
                        nodes_to_keep = list(G1.nodes())
                        to_keep = features[nodes_to_keep]
                        for n in nodes_to_keep:
                            G1.nodes[n]["x"] = features[n].argmax(-1)

                                        
                        # Assign tentative class to local explanation
                        node_attr = nx.get_node_attributes(G1,"x")
                        no_ego_attrs = set()
                        for _ , v in node_attr.items():
                            if v.item() != 4: #exclude EGO
                                no_ego_attrs.add(v.item())
                        if priori_annotation:
                            # Original version
                            if no_ego_attrs == set([0]): # MED
                                le_classes.append(0) 
                            elif no_ego_attrs == set([3]): # NUR
                                le_classes.append(1)
                            elif no_ego_attrs == set([2, 3]): # NA
                                le_classes.append(2)
                            elif no_ego_attrs == set([1]): # PAT
                                le_classes.append(3)
                            elif no_ego_attrs == set([1, 3]): # NUR + PAT
                                le_classes.append(4)
                            elif no_ego_attrs == set([0, 3]): # MED + NUR
                                le_classes.append(5)
                            elif no_ego_attrs == set([0, 1]): # MED + PAT
                                le_classes.append(6)
                            else:
                                others[str(no_ego_attrs)] += 1
                                le_classes.append(7)
                        else:
                            names = "".join(sorted(convert_labels_to_names(no_ego_attrs)))
                            if names in all_names:
                                le_classes.append(all_names[names])
                            else:
                                all_names[names] = len(all_names.keys())
                        
                        added += 1
                        adjs.append(nx.to_numpy_matrix(G1))
                        edge_weights.append(nx.get_edge_attributes(G1,"weight"))    
                        belonging.append(num_iter)
                        nodes_kept.append(nodes_to_keep)
                        ori_idxs.append(graph_id)
                        precomputed_embeddings.append(to_keep)
                if added:
                    g = nx.from_numpy_array(adj)
                    num_iter += 1
                    ori_adjs.append(adj)
                    ori_embeddings.append(features)
                    ori_edge_weights.append(nx.get_edge_attributes(g,"weight"))
                    ori_classes.append(gnn_pred)
                cont_num_iter += 1       
    belonging = utils.normalize_belonging(belonging)
    return adjs , edge_weights , ori_adjs , ori_classes , belonging , summary_predictions , le_classes, precomputed_embeddings
