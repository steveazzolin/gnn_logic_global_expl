import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx, to_networkx
import numpy as np
import networkx as nx
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt

class LocalExplanationsDataset(InMemoryDataset):
    """
        PyG Dataset object containing all disconnected local explanations
    """
    def __init__(self, root, adjs, feature_type, belonging, y=None, task_y=None, precomputed_embeddings=None, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        data_list = []
        for i , adj in enumerate(adjs):
            if feature_type == "same":
                features = torch.full((adj.shape[0], 5), 0.1)
            elif feature_type == "weights_sum": #for each node sum the edge weights
                G = nx.from_numpy_matrix(adj)
                weights = nx.get_edge_attributes(G, 'weight')
                features = torch.zeros(len(G.nodes()))
                for n in G.nodes():
                    neigh = list(G.neighbors(n))
                    total = 0
                    for p in neigh:
                        if (n,p) in weights:
                            total += weights[(n,p)]
                        else:
                            total += weights[(p,n)]
                    features[n] = total
                features = features.reshape(-1, 1)
            elif feature_type == "features":
                G = nx.from_numpy_matrix(adj)
                features = torch.zeros((len(G.nodes()), 5))
                
                d = nx.density(G)
                clu = np.mean(list(nx.clustering(G).values()))
                bet = np.mean(list(nx.betweenness_centrality(G).values()))
                clo = np.mean(list(nx.closeness_centrality(G).values()))
                emb = [d,clu,bet,clo]
                for n in G.nodes():
                    features[n, 0] = G.degree[n]
                    for q in range(len(emb)):
                        features[n, q+1] = emb[q]
            elif feature_type == "embeddings":
                features = torch.tensor(precomputed_embeddings[i])
            
            t = from_networkx(nx.from_numpy_matrix(adj))
            data = Data(x=features, 
                        edge_index=t.edge_index, 
                        edge_attr=torch.tensor(t.weight).reshape(-1, 1),
                        num_nodes=adj.shape[0],
                        y=torch.tensor(int(y[i]), dtype=torch.long) if y is not None else None, # the type of local explanation
                        task_y=torch.tensor(int(task_y[belonging[i]]), dtype=torch.long) if y is not None else None, # the class of the original input graph
                        le_id=torch.tensor(i, dtype=torch.long),
                        graph_id=belonging[i])
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.data, self.slices = self.collate(data_list)



class GroupBatchSampler():
    """
        A custom Torch sampler in order to sample in the same batch all disconnected local explanations belonging to the same input sample
    """
    def __init__(self, num_input_graphs, drop_last, belonging):
        self.batch_size = num_input_graphs
        self.drop_last = drop_last
        self.belonging = belonging
        self.num_input_graphs = num_input_graphs
        
        torch.manual_seed(42)
        random.seed(42)

    def __iter__(self):
        batch = []
        num_graphs_added = 0
        graph_idxs = random.sample(np.unique(self.belonging).tolist(), len(np.unique(self.belonging)))
        for graph_id in graph_idxs:
            le_idxs = np.where(self.belonging == graph_id)[0]
            batch.extend(le_idxs.tolist())
            if num_graphs_added >= self.batch_size:
                yield batch
                batch = []
                num_graphs_added = 0
            num_graphs_added += 1
        
        if len(batch) > 1 and not self.drop_last:
            yield batch

    def __len__(self):
        length = len(self.belonging) // self.batch_size
        if self.drop_last: return length
        else: return length + 1


class EarlyStopping():    
    def __init__(self, min_delta = 0, patience = 0):        
        self.min_delta = min_delta
        self.patience = patience
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf
        self.best_epoch = 0
        self.stop_training = False
    
    def on_epoch_end(self, epoch, current_value):
        if np.less((current_value + self.min_delta), self.best):
            self.best = current_value
            self.best_epoch = epoch
            self.wait = 0
        else:
            self.wait += 1
            if self.wait > self.patience:
                self.stopped_epoch = epoch
                self.stop_training = True
        return self.stop_training



def build_dataloader(dataset, belonging, num_input_graphs):
    batch_sampler = GroupBatchSampler(num_input_graphs=num_input_graphs, drop_last=False, belonging=np.array(belonging))
    return DataLoader(dataset, batch_sampler=batch_sampler)


def pairwise_dist(embeddings, squared=False):
    dot_prod = embeddings.mm(embeddings.T)
    square_norm = dot_prod.diag()
    dist = square_norm.unsqueeze(0) - 2*dot_prod + square_norm.unsqueeze(1) # Pairwise distances
    dist = dist.clamp(min=0) # Some values might be negative due to numerical instability. Set distences to >=0
    if not squared:
        mask = (dist==0).float() # 1 in the positions where dist==0, otherwise 0
        dist = dist + 1e-16 * mask  # Because the gradient of sqrt is infinite when dist==0, add a small epsilon where dist==0
        dist = torch.sqrt(dist)
        dist = dist * (1 - mask) # Correct the epsilon added: set the distances on the mask to be 0
    return dist

def inverse(x):
    x = 1 / (x+0.0000001)
    return x / x.sum(-1).unsqueeze(1)

def prototype_assignement(assign_func, le_embeddings, prototype_vectors, temp):
    """
        Convert the sample-prototype distance into a similarity metric/probability distribution
    """
    if assign_func == "softmax*10":
        le_assignments = F.softmax(-torch.nn.functional.normalize(torch.cdist(le_embeddings, prototype_vectors, p=2))*10 , dim=-1)
    elif assign_func == "1/x":
        le_assignments = inverse(torch.cdist(le_embeddings, prototype_vectors, p=2)) # 1/x
    elif assign_func == "sim": #from ProtoPNet
        dist = torch.cdist(le_embeddings, prototype_vectors, p=2)**2
        sim = torch.log((dist + 1) / (dist + 1e-6))
        le_assignments = F.softmax(sim/temp , dim=-1) 
    elif assign_func == "gumbel":
        dist = torch.cdist(le_embeddings, prototype_vectors, p=2)**2
        sim = torch.log((dist + 1) / (dist + 1e-6))
        le_assignments = F.gumbel_softmax(sim, tau=temp, hard=True)
    elif assign_func == "discrete":
        dist = torch.cdist(le_embeddings, prototype_vectors, p=2)**2
        sim = torch.log((dist + 1) / (dist + 1e-6))
        y_soft = F.softmax(sim / temp , dim=-1)
        
        # reparametrization trick from Gumbel Softmax
        index = y_soft.max(-1, keepdim=True)[1]
        y_hard = torch.zeros_like(y_soft).scatter_(-1, index, 1.0)
        le_assignments = y_hard - y_soft.detach() + y_soft
    return le_assignments

def entropy_loss(logits, return_raw=False):
    logp = torch.log(logits + 0.0000000001)
    entropy = torch.sum(-logits * logp, dim=1)
    if not return_raw:
        entropy = torch.mean(entropy)
    return entropy

def focal_loss(logits, targets, gamma, alpha):
    p = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    loss = loss.mean()
    return loss

def BCEWithLogitsLoss(logits, targets, gamma, alpha):
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss

def CEWithLogitsLoss(logits, targets, gamma, alpha):
    loss = F.cross_entropy(logits, targets)
    return loss

@torch.no_grad()
def get_cluster_accuracy(concept_predictions, classes):
    accs = []
    for cl in np.unique(concept_predictions):
        _ , counts = np.unique(classes[concept_predictions == cl], return_counts=True)            
        accs.append(np.max(counts) / np.sum(counts))
    return accs

def normalize_belonging(belonging):
    ret = []
    i = -1
    for j , elem in enumerate(belonging):
        if len(ret) > 0 and elem == belonging[j-1]:
            ret.append(i)
        else:
            i += 1
            ret.append(i)
    return ret

def rewrite_formula_to_close(formula):
    """
    Takes a formula written with open world assumption and converts to closed one, i.e,
    remove every negated literal assuming that missing literals are by default negated
    """
    ret = []
    for clause in formula.split("|"):
        tmp = "("
        for f in clause.split("&"):
            if "~" not in f:
                if tmp != "(":
                    tmp += " & "
                tmp += f.strip().replace(")", "")
        tmp += ")"
        ret.sort(key=lambda x: len(x))
        ret.append(tmp)
    return " | ".join(ret)

def assemble_raw_explanations(explanations_raw):
    ret = []
    for expl in explanations_raw:
        ret.append(f"({expl})")
    return " | ".join(ret)

def plot_molecule(data, adj=None, node_features=None, composite_plot=False):
    if adj is None and node_features is None:
        G = to_networkx(data)
    else:
        G = nx.from_numpy_matrix(adj)
    node_label = data.x.argmax(-1) if node_features is None else node_features.argmax(-1)
    max_label = node_label.max() + 1
    nmb_nodes = len(node_label)
    
    colors = ['orange','red','lime','green','lightseagreen','orchid','darksalmon','darkslategray','gold','bisque','tan','blue','indigo','navy']
    color_to_atom = {
        'orange': "C",
        'red': "O",
        'lime': "Cl",
        'green': "H",
        'lightseagreen': "N",
        'orchid': "F",
        'darksalmon': "Br",
        'darkslategray': "S",
        'gold': "P",
        'bisque': "I",
        'tan': "Na",
        'blue': "K",
        'indigo': "Li",
        'navy': "Ca"
    }

    label2nodes = []
    for i in range(max_label):
        label2nodes.append([])
    for i in range(nmb_nodes):
        if i in G.nodes():
            label2nodes[node_label[i]].append(i)
            
    if adj is not None and node_features is not None:
        G.remove_edges_from(nx.selfloop_edges(G))
        G.remove_nodes_from(list(nx.isolates(G)))
            
    pos = nx.kamada_kawai_layout(G)
    for i in range(max_label):
        node_filter = []
        for j in range(len(label2nodes[i])):
            node_filter.append(label2nodes[i][j])
        nx.draw_networkx_nodes(G, pos,
                               nodelist=node_filter,
                               node_color=colors[i],
                               node_size=300)
        nx.draw_networkx_labels(G, pos, {k:color_to_atom[colors[i]] for k in node_filter})

    nx.draw_networkx_edges(G, pos, width=2, edge_color='grey')
    plt.box(False)

    if not composite_plot:
        plt.axis('off')
        plt.show()


def convert_hin_labels(g):
    new_dict = dict()
    for n,x in dict(nx.get_node_attributes(g,"x")).items():
        if x[0]==1:
            new_dict[n]="D"
        if x[1]==1:
            new_dict[n]="P"
        if x[2]==1:
            new_dict[n]="A"
        if x[3]==1:
            new_dict[n]="N"
        if x[4]==1:
            new_dict[n]="Ego"
    return new_dict

def plot_etn(data, ax=None):
    G = to_networkx(data, node_attrs=["x"], to_undirected=True)
    pos = nx.kamada_kawai_layout(G)
    nx.draw(G, pos, node_color="orange", node_size=400, ax=ax)
    nx.draw_networkx_labels(G, pos, {k:v for k,v in convert_hin_labels(G).items() if k in G.nodes()}, font_size=18, font_color="black", ax=ax)