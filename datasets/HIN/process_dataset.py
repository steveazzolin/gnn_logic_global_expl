import pandas as pd
import pickle as pkl
import numpy as np
import networkx as nx


def load_raw_hin(gap):
    # read file from http://www.sociopatterns.org/datasets/hospital-ward-dynamic-contact-network/
    data = pd.read_csv("sociopattern_data/lh.dat", sep="\t", names=["time","A","B","fa","fb"])
    tmp = data.time.to_numpy()
    tmp = tmp - min(tmp)

    # extract metadata
    data_to_class = dict()
    for _ , person_data in data.iterrows():
        src , dest , src_type , dest_type = person_data[1:]
        data_to_class[src] = src_type
        data_to_class[dest] = dest_type    

    # create temporal snapshots
    tmp = tmp // gap
    a = np.unique(tmp)
    dict_a = {}
    c = 0
    for i in a:    
        dict_a[i] = c
        c = c + 1

    new_tmp = []
    for i in tmp:
        new_tmp.append(dict_a[i])
    data.time  = new_tmp    

    c = 0
    snapshots = []
    for _ , snap in data.groupby("time"):
        G = nx.Graph()
        edge_list = snap.to_numpy()[:,1:3]
        G.add_edges_from(edge_list)
        for n in G.nodes():
            G.nodes()[n]["lab"] = data_to_class[n]

        snapshots.append(G.copy())
    return snapshots

def split_by_category(graphs, radius, min_nodes, label, set_ego_label=True):
    ret = []
    for G in graphs:
        for n,l in dict(nx.get_node_attributes(G, "lab")).items():
            if l == label:
                gg = nx.ego_graph(G, n, radius=radius)
                if len(gg) >= min_nodes:                    
                    gg.nodes()[n]["lab"] = "UNK"
                    ret.append(gg)
    return ret

def extract_summarization_metrics(graphs):
    n = []
    e = []
    d = []
    for i in graphs:
        n.append(len(i.nodes()))
        e.append(len(i.edges()))
        d.append(nx.density(i))        
    return np.mean(n) , np.mean(e) , np.mean(d)

def get_node_feature_from_type(node_type):
    if node_type == "MED":
        return [1,0,0,0,0]
    elif node_type == "NUR":
        return [0,0,0,1,0]
    elif node_type == "PAT":
        return [0,1,0,0,0]
    elif node_type == "ADM":
        return [0,0,1,0,0]
    elif node_type == "UNK":
        return [0,0,0,0,1]

def create_refined_hin(graphs):
    adjs , feas , labels = [] , [] , []    
    min_cout_samples = min([len(g) for g in graphs])
    for i , category in enumerate(graphs):
        for g in category[:min_cout_samples]:
            fea = []
            for i,j in nx.get_node_attributes(g,"lab").items():
                fea.append(get_node_feature_from_type(j))
            feas.append(fea.copy())
            adjs.append(nx.adjacency_matrix(g).A)
            labels.append(i)
    return adjs , feas, labels

def create_and_save_dataset():
    graphs = load_raw_hin(gap=60*5)
    graphs_med = split_by_category(graphs, radius=3, min_nodes=7, label="MED")
    graphs_nur = split_by_category(graphs, radius=3, min_nodes=7, label="NUR")

    print("Num graphs med: ", len(graphs_med), "Num graphs nur: ",  len(graphs_nur))
    #print(extract_summarization_metrics(graphs_med))
    #print(extract_summarization_metrics(graphs_nur))
    
    adjs , feas, labels = create_refined_hin([graphs_nur, graphs_med])
    f = open('HIN3.pkl','wb')
    pkl.dump((adjs,feas,labels),f)
    f.close()



if __name__ == "__main__":
    create_and_save_dataset()