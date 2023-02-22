import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, global_mean_pool, global_add_pool, global_max_pool, GINConv, GATv2Conv, GraphConv
from torch_scatter import scatter
import torch_explain as te
from torch_explain.logic.nn import entropy
from torch_explain.logic.metrics import test_explanation, test_explanations
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from scipy.stats import hmean
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import wandb
import time
import utils




class GLGExplainer(torch.nn.Module):
    """
        Implementation of GLGExplainer (https://arxiv.org/abs/2210.07147)
    """
    def __init__(self, len_model, le_model, device, hyper_params, classes_names, dataset_name, num_classes):
        super().__init__()        
        
        self.le_model = le_model
        self.len_model = len_model
        
        self.prototype_vectors = torch.nn.Parameter(
            torch.rand((hyper_params["num_prototypes"], 
                        hyper_params["dim_prototypes"])), requires_grad=True)       
        
        self.train_metrics , self.val_metrics , self.train_logic_metrics , self.val_logic_metrics = [] , [] , [] , []
        self.device = device
        self.hyper = hyper_params
        self.classes_names = classes_names
        self.num_classes = num_classes
        self.dataset_name = dataset_name
        self.temp = hyper_params["ts"]
        self.assign_func = hyper_params["assign_func"]
        self.early_stopping = utils.EarlyStopping(min_delta=0, patience=100)
        self.losses_names = ["loss", "prototype_distance_loss", "r1_loss", "r2_loss", "concept_entropy_loss", "distribution_entropy_loss", "div_loss", "len_loss", "debug_loss", "logic_loss"]
        
        self.optimizer = torch.optim.Adam(le_model.parameters(), lr=self.hyper["le_emb_lr"])
        self.optimizer.add_param_group({'params': len_model.parameters(), 'lr': self.hyper["len_lr"]})
        self.optimizer.add_param_group({'params': self.prototype_vectors, 'lr': self.hyper["proto_lr"]})
        
        if hyper_params["focal_loss"]:
            self.loss_len = utils.focal_loss
        else:
            if self.num_classes == 2:
                self.loss_len = utils.BCEWithLogitsLoss
            elif self.num_classes == 3:
                self.loss_len = utils.CEWithLogitsLoss
            else:
                raise NotImplementedError("num_classes implemented <= 3")
        
        
    def get_concept_vector(self, loader, return_raw=False):
        le_embeddings = torch.tensor([], device=self.device)
        new_belonging = torch.tensor([], device=self.device, dtype=torch.long)
        y = torch.tensor([], device=self.device)
        le_classes = torch.tensor([], device=self.device)
        le_idxs = torch.tensor([], device=self.device)
        for data in loader:
            data = data.to(self.device)
            le_idxs = torch.concat([le_idxs, data.le_id], dim=0)
            embs = self.le_model(data.x, data.edge_index, data.batch)
            le_embeddings = torch.concat([le_embeddings, embs], dim=0)
            new_belonging = torch.concat([new_belonging, data.graph_id], dim=0)
            le_classes = torch.concat([le_classes, data.y], dim=0)
            y = torch.concat([y, data.task_y], dim=0)
            
        y = scatter(y, new_belonging, dim=0, reduce="max")
        y = torch.nn.functional.one_hot(y.long()).float().to(self.device)
             
        le_assignments = utils.prototype_assignement(self.hyper["assign_func"], le_embeddings, self.prototype_vectors, temp=1)        
        concept_vector = scatter(le_assignments, new_belonging, dim=0, reduce="max")
        if return_raw:
            return concept_vector , le_embeddings ,  le_assignments , y , le_classes.cpu() , le_idxs , new_belonging
        else:
            return concept_vector , le_embeddings

    
    def train_epoch(self, loader, train=True):   
        if train:
            self.train()
        else:
            self.eval()
    
        total_losses = {k: torch.tensor(0., device=self.device) for k in self.losses_names}        
        preds , trues = torch.tensor([], device=self.device) , torch.tensor([], device=self.device)
        le_classes = torch.tensor([], device=self.device)
        total_prototype_assignements = torch.tensor([], device=self.device)

        for data in loader:
            self.optimizer.zero_grad() 
            data = data.to(self.device)
            le_embeddings = self.le_model(data.x, data.edge_index, data.batch)
            
            new_belonging = torch.tensor(utils.normalize_belonging(data.graph_id), dtype=torch.long, device=self.device)
            y = scatter(data.task_y, new_belonging, dim=0, reduce="max")
            y_train_1h = torch.nn.functional.one_hot(y.long(), num_classes=self.num_classes).float().to(self.device)
            
            prototype_assignements = utils.prototype_assignement(self.hyper["assign_func"], le_embeddings, self.prototype_vectors, temp=1)
            total_prototype_assignements = torch.cat([total_prototype_assignements, prototype_assignements], dim=0)
            le_classes = torch.concat([le_classes, data.y], dim=0)
            concept_vector = scatter(prototype_assignements, new_belonging, dim=0, reduce="max")
            
            loss , y_pred = self.compute_losses(le_embeddings, prototype_assignements, total_losses, concept_vector, y_train_1h, data.y)
            preds = torch.cat([preds, y_pred], dim=0)
            trues = torch.cat([trues, y_train_1h], dim=0)
            
            if loss is None:
                continue
            
            if train:
                loss.backward()
                self.optimizer.step()      
        
        if self.hyper["debug_prototypes"]:
            acc_per_class = 0
            acc_overall = accuracy_score(trues.cpu(), preds.argmax(-1).cpu())
        else:
            acc_per_class = accuracy_score(trues.argmax(-1).cpu(), preds.argmax(-1).cpu()) 
            acc_overall   = sum(trues[:, :].eq(preds[:, :] > 0).sum(1) == self.num_classes) / len(preds) # it checks that the LEN predicted only one class. acc_per_class instead consider a sample correct even if the LEN fires both classes
        
        cluster_acc = utils.get_cluster_accuracy(
            total_prototype_assignements.argmax(1).detach().cpu().numpy(), 
            le_classes.cpu())

        metrics                           = {k: v.item() / len(loader) for k , v in total_losses.items()}
        metrics["acc_per_class"]          = acc_per_class
        metrics["acc_overall"]            = acc_overall
        metrics["temperature"]            = self.temp
        metrics["cluster_acc_mean"]       = np.mean(cluster_acc)
        metrics["cluster_acc_std"]        = np.std(cluster_acc)
        metrics["concept_vector_entropy"] = utils.entropy_loss(prototype_assignements).detach().cpu() # change to total_prototype_assignements for the full information (more expensive)
        metrics["prototype_assignements"] = wandb.Histogram(prototype_assignements.detach().cpu())
        metrics["concept_vector"]         = wandb.Histogram(concept_vector.detach().cpu())         
            
        if self.hyper["log_wandb"]:
            k = "train" if train else "val"
            self.log({k: metrics}) 
        else:
            if train:
                self.train_metrics.append(metrics)
            else:
                self.val_metrics.append(metrics)
        return metrics


    def debug_prototypes(self, le_embeddings, prototype_assignements, y):
        debug_loss = 1 * F.cross_entropy(prototype_assignements, y)

        sample_prototype_distance = torch.cdist(le_embeddings, self.prototype_vectors, p=2)**2 # num_sample x num_prototypes
        min_prototype_sample_distance = sample_prototype_distance.T.min(-1).values
        avg_prototype_sample_distance = torch.mean(min_prototype_sample_distance)
        r1_loss = self.hyper["coeff_r1"] * avg_prototype_sample_distance     

        min_sample_prototype_distance = sample_prototype_distance.min(-1).values
        avg_sample_prototype_distance = torch.mean(min_sample_prototype_distance)
        r2_loss = self.hyper["coeff_r2"] * avg_sample_prototype_distance

        loss = debug_loss +  r1_loss + r2_loss 
        loss.backward()
        self.optimizer.step()
        return loss, r1_loss , r2_loss , debug_loss
        
        
    def iterate(self, train_loader, val_loader, name_wandb="", config_wandb=None, save_metrics=True, plot=False):
        if self.hyper["log_wandb"]:
            self.run = wandb.init(
                    project=config_wandb["project_name"],
                    name=name_wandb,
                    entity=config_wandb["entity_name"],
                    reinit=config_wandb["reinit"],
                    save_code=config_wandb["save_code"],
                    config=self.hyper
            )
            wandb.watch(self.le_model)
            wandb.watch(self.len_model)        
        
        if plot: 
            self.inspect(train_loader)        

        start_time = time.time()
        best_val_loss = np.inf
        for epoch in range(1, self.hyper["num_epochs"]):
            train_metrics = self.train_epoch(train_loader)
            val_metrics   = self.train_epoch(val_loader, train=False)
            
            if epoch % 20 == 0:
                self.inspect(train_loader, self.hyper["log_wandb"], plot=plot)
                self.inspect(val_loader, log_wandb=False, plot=False, is_train_set=False)
                
            self.temp -= (self.hyper["ts"] - self.hyper["te"]) / self.hyper["num_epochs"]
            if self.hyper["log_wandb"] and self.hyper["log_models"]:
                torch.save(self.state_dict(), f"{wandb.run.dir}/epoch_{epoch}.pt")  
            if val_metrics["loss"] < best_val_loss and self.hyper["log_models"]:
                best_val_loss = val_metrics["loss"]
                torch.save(self.state_dict(), f"../trained_models/best_so_far/best_so_far_{self.dataset_name}_epoch_{epoch}.pt")

            print(f'{epoch:3d}: Loss: {train_metrics["loss"]:.5f}, LEN: {train_metrics["len_loss"]:2f}, Acc: {train_metrics["acc_overall"]:.2f}, V. Acc: {val_metrics["acc_overall"]:.2f}, V. Loss: {val_metrics["loss"]:.5f}, V. LEN {val_metrics["len_loss"]:.3f}')
                
            if self.early_stopping.on_epoch_end(epoch, val_metrics["loss"]):
                print(f"Early Stopping")
                print(f"Loading model at epoch {self.early_stopping.best_epoch}")
                if self.hyper["log_models"]:
                    self.load_state_dict(torch.load(f"../trained_models/best_so_far/best_so_far_{self.dataset_name}_epoch_{self.early_stopping.best_epoch}.pt"))
                else:
                    print("Model not loaded")
                break
        print(f"Best epoch: {self.early_stopping.best_epoch}")   
        print(f"Trained lasted for {round(time.time() - start_time)} seconds")
                
        if self.hyper["log_wandb"]:
            if self.hyper["log_models"]:
                wandb.save(f'{wandb.run.dir}/epoch_*.pt')
            self.run.finish()  
        
        # if save_metrics:
        #     with open(f'../logs/ablation/num_proto/{self.dataset_name}/{self.hyper["num_prototypes"]}_train_metrics.pkl', 'wb') as handle:
        #         pickle.dump(self.train_metrics, handle)
        #     with open(f'../logs/ablation/num_proto/{self.dataset_name}/{self.hyper["num_prototypes"]}_val_metrics.pkl', 'wb') as handle:
        #         pickle.dump(self.val_metrics, handle)        
        #     with open(f'../logs/ablation/num_proto/{self.dataset_name}/{self.hyper["num_prototypes"]}_train_logic_metrics.pkl', 'wb') as handle:
        #         pickle.dump(self.train_logic_metrics, handle)        
        #     with open(f'../logs/ablation/num_proto/{self.dataset_name}/{self.hyper["num_prototypes"]}_val_logic_metrics.pkl', 'wb') as handle:
        #         pickle.dump(self.val_logic_metrics, handle)     
        return

    
    def inspect(self, loader, log_wandb=False, plot=True, is_train_set=False):
        self.eval()
        
        with torch.no_grad():
            x_train , emb , concepts_assignement , y_train_1h , le_classes , le_idxs , belonging = self.get_concept_vector(loader, return_raw=True)        
            y_pred = self.len_model(x_train).squeeze(-1).cpu()

            emb = emb.detach().cpu().numpy()
            concept_predictions = concepts_assignement.argmax(1).cpu().numpy()
        
            if plot: # plot embedding                
                pca = PCA(n_components=2, random_state=42)
                emb2d = emb if self.prototype_vectors.shape[1] == 2 else pca.fit_transform(emb) #emb
                fig = plt.figure(figsize=(17,4))
                plt.subplot(1,2,1)
                plt.title("local explanations embeddings", size=23)
                print(np.unique(le_classes, return_counts=True))
                for c in np.unique(le_classes):
                    plt.scatter(emb2d[le_classes == c,0], emb2d[le_classes == c,1], label=self.classes_names[int(c)], alpha=0.7)
                proto_2d = self.prototype_vectors.cpu().numpy() if self.prototype_vectors.shape[1] == 2 else pca.transform(self.prototype_vectors.cpu().numpy())
                plt.scatter(proto_2d[:, 0], proto_2d[:,1], marker="x", s=60, c="black")        
                for i, txt in enumerate(range(proto_2d.shape[0])):
                    plt.annotate("p" + str(i), (proto_2d[i,0]+0.01, proto_2d[i,1]+0.01), size=27)
                plt.legend(bbox_to_anchor=(0.04,1), prop={'size': 17})
                plt.subplot(1,2,2)
                plt.title("prototype assignments", size=23)
                for c in range(self.prototype_vectors.shape[0]):
                    plt.scatter(emb2d[concept_predictions == c,0], emb2d[concept_predictions == c,1], label="p"+str(c))
                plt.legend(prop={'size': 17})

                # plt.subplot(1,3,3)
                # plt.title("predictions")
                # idx_belonging_correct = y_train_1h[:, :].eq(y_pred[:, :] > 0).sum(1) == 2 #y_pred.argmax(1) == y_train_1h.argmax(1)
                # idx_belonging_wrong   = y_train_1h[:, :].eq(y_pred[:, :] > 0).sum(1) != 2
                # colors = []
                # for idx in range(emb2d.shape[0]):
                #     if idx_belonging_correct[belonging[idx]]:
                #         colors.append("blue")
                #     elif idx_belonging_wrong[belonging[idx]]:
                #         colors.append("red")
                # plt.scatter(emb2d[:, 0], emb2d[:, 1], c=colors)
                # patches = [mpatches.Patch(color='blue', label='correct'), mpatches.Patch(color='red', label='wrong')]
                # plt.legend(handles=patches)

                if log_wandb and self.hyper["log_images"]: 
                    wandb.log({"plots": wandb.Image(plt)})
                if self.prototype_vectors.shape[1] > 2: print(pca.explained_variance_ratio_)
                fig.supxlabel('principal comp. 1', size=20)
                fig.supylabel('principal comp. 2', size=20)                
                #plt.savefig("embedding_bamultishapesmc.pdf")
                plt.show()         


            #log stats
            if isinstance(self.len_model[0], te.nn.logic.EntropyLinear) and plot:
                print("Alpha norms:")
                print(self.len_model[0].alpha_norm)
            
            self.len_model.to("cpu")
            x_train = x_train.detach().cpu()
            y_train_1h = y_train_1h.cpu()
            explanation0, explanation_raw = entropy.explain_class(self.len_model, x_train, y_train_1h, train_mask=torch.arange(x_train.shape[0]).long(), val_mask=torch.arange(x_train.shape[0]).long(), target_class=0, max_accuracy=True, topk_explanations=3000, try_all=False)
            accuracy0, preds = test_explanation(explanation0, x_train, y_train_1h, target_class=0, mask=torch.arange(x_train.shape[0]).long(), material=False)
            
            cluster_accs = utils.get_cluster_accuracy(concept_predictions, le_classes)
            if plot:
                print(f"Concept Purity: {np.mean(cluster_accs):2f} +- {np.std(cluster_accs):2f}")
                print("Concept distribution: ", np.unique(concept_predictions, return_counts=True))        
                print("Logic formulas:")
                print("For class 0:")
                print(accuracy0, utils.rewrite_formula_to_close(utils.assemble_raw_explanations(explanation_raw)))

            explanation1, explanation_raw = entropy.explain_class(self.len_model, x_train, y_train_1h, train_mask=torch.arange(x_train.shape[0]).long(), val_mask=torch.arange(x_train.shape[0]).long(), target_class=1, max_accuracy=True, topk_explanations=3000, try_all=False)
            accuracy1, preds = test_explanation(explanation1, x_train, y_train_1h, target_class=1, mask=torch.arange(x_train.shape[0]).long(), material=False)
            
            if plot:
                print("For class 1:")
                print(accuracy1, utils.rewrite_formula_to_close(utils.assemble_raw_explanations(explanation_raw)))

            if self.num_classes == 3:
                explanation2, explanation_raw = entropy.explain_class(self.len_model, x_train, y_train_1h, train_mask=torch.arange(x_train.shape[0]).long(), val_mask=torch.arange(x_train.shape[0]).long(), target_class=2, max_accuracy=True, topk_explanations=3000, try_all=False)
                accuracy2, preds = test_explanation(explanation2, x_train, y_train_1h, target_class=2, mask=torch.arange(x_train.shape[0]).long(), material=False)
                if plot:
                    print("For class 2:")
                    print(accuracy2, utils.rewrite_formula_to_close(utils.assemble_raw_explanations(explanation_raw)))
                accuracy, preds = test_explanations([explanation0, explanation1, explanation2], x_train, y_train_1h, mask=torch.arange(x_train.shape[0]).long(), material=False)
                logic_acc = hmean([accuracy0, accuracy1, accuracy2])
            else:
                accuracy, preds = test_explanations([explanation0, explanation1], x_train, y_train_1h, mask=torch.arange(x_train.shape[0]).long(), material=False)
                logic_acc = hmean([accuracy0, accuracy1])

            if plot: print("Accuracy as classifier: ", round(accuracy, 4))
            if plot: print("LEN fidelity: ", sum(y_train_1h[:, :].eq(y_pred[:, :] > 0).sum(1) == self.num_classes) / len(y_pred))
            
            print()
            if log_wandb: self.log({"train": {'logic_acc': logic_acc, "logic_acc_clf": accuracy}})
            else: 
                if is_train_set:
                    self.train_logic_metrics.append({'logic_acc': logic_acc, "logic_acc_clf": accuracy, "concept_purity": np.mean(cluster_accs), "concept_purity_std": np.std(cluster_accs)})
                else:
                    self.val_logic_metrics.append({'logic_acc': logic_acc, "logic_acc_clf": accuracy, "concept_purity": np.mean(cluster_accs), "concept_purity_std": np.std(cluster_accs)})
        self.len_model.to(self.device)


    def compute_losses(self, le_embeddings, prototype_assignements, total_losses, concept_vector, y_train_1h, le_y):
        # debug of prototypes: do direct classification on prototypes
        if self.hyper["debug_prototypes"]:                
            loss , r1_loss , r2_loss , debug_loss = self.debug_prototypes(le_embeddings, prototype_assignements, le_y)
            total_losses["loss"] += loss.detach()
            total_losses["r1_loss"] += r1_loss.detach()
            total_losses["r2_loss"] += r2_loss.detach()
            total_losses["debug_loss"] += debug_loss.detach()            
            #preds = torch.cat([preds, prototype_assignements], dim=0)
            #trues = torch.cat([trues, le_y], dim=0)
            return None , prototype_assignements
            
        # LEN clf. loss
        y_pred = self.len_model(concept_vector).squeeze(-1)
        #preds = torch.cat([preds, y_pred], dim=0)
        #trues = torch.cat([trues, y_train_1h], dim=0)
        len_loss = 0.5 * self.loss_len(y_pred, y_train_1h, self.hyper["focal_gamma"], self.hyper["focal_alpha"])
        total_losses["len_loss"] += len_loss.detach()   


        # R1 loss: push each prototype to be close to at least one example
        if self.hyper["coeff_r1"] > 0:
            sample_prototype_distance = torch.cdist(le_embeddings, self.prototype_vectors, p=2)**2 # num_sample x num_prototypes
            min_prototype_sample_distance = sample_prototype_distance.T.min(-1).values
            avg_prototype_sample_distance = torch.mean(min_prototype_sample_distance)
            r1_loss = self.hyper["coeff_r1"] * avg_prototype_sample_distance
            total_losses["r1_loss"] += r1_loss.detach()
        else:
            r1_loss = torch.tensor(0., device=self.device)

        # R2 loss: Push every example to be close to a sample
        if self.hyper["coeff_r2"] > 0:
            sample_prototype_distance = torch.cdist(le_embeddings, self.prototype_vectors, p=2)**2
            min_sample_prototype_distance = sample_prototype_distance.min(-1).values
            avg_sample_prototype_distance = torch.mean(min_sample_prototype_distance)
            r2_loss = self.hyper["coeff_r2"] * avg_sample_prototype_distance
            total_losses["r2_loss"] += r2_loss.detach()
        else:
            r2_loss = torch.tensor(0., device=self.device)         
        
        # Logic loss defined by Entropy Layer
        if self.hyper["coeff_logic_loss"] > 0:
            logic_loss = self.hyper["coeff_logic_loss"] * te.nn.functional.entropy_logic_loss(self.len_model)
            total_losses["logic_loss"] += logic_loss.detach()
        else:
            logic_loss = torch.tensor(0., device=self.device)            
        
        # Prototype distance: push away the different prototypes by maximizing the distance to the nearest prototype
        if self.hyper["coeff_pdist"] > 0:
            prototype_distances = torch.clip(utils.pairwise_dist(self.prototype_vectors), max=0.5).fill_diagonal_(float("inf"))
            prototype_distances = prototype_distances.min(-1).values
            prototype_distance_loss = self.hyper["coeff_pdist"] * - torch.mean(prototype_distances)
            total_losses["prototype_distance_loss"] += prototype_distance_loss.detach()
        else:
            prototype_distance_loss = torch.tensor(0., device=self.device)
            
        # Div loss: from ProtGNN minimize the cosine similarity between prototypes with some margin
        if self.hyper["coeff_divloss"] > 0:
            proto_norm = F.normalize(self.prototype_vectors, p=2, dim=1)
            cos_distances = torch.mm(proto_norm, torch.t(proto_norm)) - torch.eye(proto_norm.shape[0]).to(self.device) - 0.2
            matrix2 = torch.zeros(cos_distances.shape).to(self.device)
            div_loss = self.hyper["coeff_divloss"] * torch.sum(torch.where(cos_distances > 0, cos_distances, matrix2))   
            total_losses["div_loss"] += div_loss.detach()
        else:
            div_loss = torch.tensor(0., device=self.device)

        # 2 Entropy losses
        if self.hyper["coeff_ce"] > 0:
            concept_entropy_loss = self.hyper["coeff_ce"] * utils.entropy_loss(prototype_assignements)
            total_losses["concept_entropy_loss"] += concept_entropy_loss.detach()
        else:
            concept_entropy_loss = torch.tensor(0., device=self.device)
            
        if self.hyper["coeff_de"] > 0:
            distribution_entropy_loss = self.hyper["coeff_de"] * utils.entropy_loss(
                torch.nn.functional.normalize(
                    torch.sum(prototype_assignements, dim=0),
                    p=2.0, dim=0).unsqueeze(0)
            ) 
            total_losses["distribution_entropy_loss"] += distribution_entropy_loss.detach()      
        else:
            distribution_entropy_loss = torch.tensor(0., device=self.device)

        loss = len_loss + logic_loss   + prototype_distance_loss + r1_loss + r2_loss + concept_entropy_loss + div_loss + distribution_entropy_loss
        total_losses["loss"] += loss.detach()
        return loss , y_pred
        
    def log(self, msg):
        wandb.log(msg)

    def eval(self):
        self.le_model.eval()
        self.len_model.eval()

    def train(self):
        self.le_model.train()
        self.len_model.train()


class LEEmbedder(torch.nn.Module):
    """
        Network for computing the embedding of single disconnected local explanations
    """
    def __init__(self, num_features, activation, num_gnn_hidden=20, dropout=0.1, num_hidden=10, num_layers=2, backbone="GIN"):
        super().__init__()

        if backbone == "GIN":
            nns = torch.nn.ModuleList([
                torch.nn.Sequential(
                    torch.nn.Linear(num_features if i == 0 else num_gnn_hidden, num_gnn_hidden),
                    torch.nn.LeakyReLU(),
                    torch.nn.Dropout(dropout)
                )
            for i in range(num_layers)
            ])
            self.convs = torch.nn.ModuleList([
                GINConv(nns[i], train_eps=False) for i in range(num_layers)
            ])
        elif backbone == "GAT":
            self.convs = torch.nn.ModuleList([
                GATv2Conv(num_features if i == 0 else num_gnn_hidden, int(num_gnn_hidden/4), heads=4) for i in range(num_layers)
            ])
        elif backbone == "SAGE":
            self.convs = torch.nn.ModuleList([
                SAGEConv(num_features if i == 0 else num_gnn_hidden, num_gnn_hidden) for i in range(num_layers)
            ])
        elif backbone == "SAGE_sum":
            self.convs = torch.nn.ModuleList([
                SAGEConv(num_features if i == 0 else num_gnn_hidden, num_gnn_hidden, aggr="add") for i in range(num_layers)
            ])
        elif backbone == "GCN":
            self.convs = torch.nn.ModuleList([
                GCNConv(num_features if i == 0 else num_gnn_hidden, num_gnn_hidden) for i in range(num_layers)
            ])
        elif backbone == "GraphConv":
            self.convs = torch.nn.ModuleList([
                GraphConv(num_features if i == 0 else num_gnn_hidden, num_gnn_hidden) for i in range(num_layers)
            ])
        else:
            raise ValueError("Backbone not available") 

        self.proj = torch.nn.Linear(num_gnn_hidden * 3, num_hidden)
        self.num_layers = num_layers

        if activation == "sigmoid":
            self.actv = torch.nn.Sigmoid()
        elif activation == "tanh":
            self.actv = torch.nn.Tanh()
        elif activation == "leaky":
            self.actv = torch.nn.LeakyReLU()
        elif activation == "lin":
            self.actv = torch.nn.LeakyReLU(negative_slope=1)
        else:
            raise ValueError("Activation not available") 


    def forward(self, x, edge_index, batch):
        x = self.get_graph_emb(x, edge_index, batch)
        x = self.actv(self.proj(x))
        return x
    
    def get_graph_emb(self, x, edge_index, batch):
        x = self.get_emb(x, edge_index)

        x1 = global_mean_pool(x, batch)
        x2 = global_add_pool(x, batch)
        x3 = global_max_pool(x, batch)
        x = torch.cat([x1, x2, x3], dim=-1)
        return x

    def get_emb(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.actv(self.convs[i](x.float(), edge_index))
        return x

    


def LEN(input_shape, temperature, n_classes=2, remove_attention=False):
    layers = [
        te.nn.EntropyLinear(input_shape, 10, n_classes=n_classes, temperature=temperature, remove_attention=remove_attention),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(10, 5),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(5, 1),
    ]
    return torch.nn.Sequential(*layers)






