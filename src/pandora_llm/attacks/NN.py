from tqdm import tqdm
import os
import math
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from .Attack import MIA
from ..utils.plot_utils import print_AUC

####################################################################################################
# MAIN CLASS
####################################################################################################
class NN(MIA):
    def __init__(self, clf_name, feature_set, clf_size, model_name, model_revision=None, model_cache_dir=None):
        self.clf_name = clf_name
        self.clf = None
        self.clf_size = clf_size
        self.scaler = None
        self.feature_set = feature_set
        self.model_name = model_name
        self.model_revision = model_revision
        self.model_cache_dir = model_cache_dir

    def preprocess_features(self, features, labels = None, fit_scaler=False):
        """
        Preprocesses features into a format accepted by the classifier.

        Args:
            features (dict[str,torch.Tensor]): dictionary of feature_name to list of features
            labels (Optional[torch.Tensor]): list of labels. If specified, returns the labels as well
            fit_scaler (Optional[bool]): whether to fit scaler or not
        Returns:
            processed_features, Optional[labels]
        """
        if self.scaler is None and not fit_scaler:
            raise Exception("Please call preprocess_features with fit_scaler=True first!")
        if self.scaler is not None and fit_scaler:
            raise RuntimeWarning("Refitting scaler! Please use the same scaler for both training and evaluation!")
        
        # Collect all features
        processed_features = []
        for feature_name in self.feature_set:
            feature = features[feature_name]
            if feature.ndim==1:
                processed_features.append(feature.reshape(-1,1))
            else:
                processed_features.append(feature)
        processed_features = torch.cat(processed_features,dim=1)
        
        # Preprocess
        processed_features = torch.nan_to_num(processed_features)
        mask_out_infinite = processed_features.isfinite().all(axis=1)
        processed_features = processed_features[mask_out_infinite]
        if fit_scaler:
            self.scaler = StandardScaler().fit(processed_features)
        processed_features = torch.from_numpy(self.scaler.transform(processed_features))
        
        # Return
        if labels is not None:
            labels = labels[mask_out_infinite]
            return processed_features, labels
        else:
            return processed_features

    def train_clf(self, train_features, train_labels, test_features, test_labels, clf_size, epochs, batch_size, patience, min_delta=0., device=None):
        """
        Take train and validation data and train neural network as MIA.

        Args:
            train_features (torch.Tensor): Features for training supervised MIA 
            train_labels (torch.Tensor): Labels for train data (binary, 1 is train)
            test_features (torch.Tensor): Features for validating supervised MIA 
            test_labels (torch.Tensor): Labels for test data (binary, 1 is train)
            clf_size (str): which neural net architecture to use
            epochs (int): number of epochs to train
            batch_size (int): batch size to train with
            patience (int): number of epochs to wait for improvement before stopping
            device (str): device to train on
                    
        Returns:
            tuple[torch.Tensor]: train predictions 
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.clf = NeuralNetwork(train_features.shape[1], clf_size).to(device)

        train_dataset = TensorDataset(train_features, train_labels)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = TensorDataset(test_features, test_labels)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        optimizer = Adam(self.clf.parameters())
        loss_fn = nn.BCELoss()

        # Training
        epochs_no_improve = 0
        best_loss = float('inf')
        for epoch in tqdm(range(epochs)):
            self.clf.train()
            total_loss = 0
            correct = 0
            total = 0
            all_probs = []
            all_labels = []

            for step, (data, target) in enumerate(train_dataloader):
                optimizer.zero_grad()
                probs = self.clf(data.to(device).float())[:,0]
                loss = loss_fn(probs, target.to(device))
                total_loss += loss.item()
                pred = (probs > 0.5).detach().cpu().int()
                correct += (pred == target).sum().item()
                total += target.size(0)
                all_probs.append(probs.detach().cpu())
                all_labels.append(target)
                loss.backward()
                optimizer.step()
            
            # Validation
            self.clf.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            all_probs_val = []
            all_labels_val = []
            with torch.no_grad():
                for data, target in test_dataloader:
                    probs = self.clf(data.to(device).float())[:, 0]
                    loss = loss_fn(probs, target.to(device))
                    val_loss += loss.item()
                    pred = (probs > 0.5).detach().cpu().int()
                    val_correct += (pred == target).sum().item()
                    val_total += target.size(0)
                    all_probs_val.append(probs.detach().cpu())
                    all_labels_val.append(target)
            
            # Early stopping
            if best_loss - val_loss > min_delta:
                best_loss = val_loss
                epochs_no_improve = 0
                # Save the model
                best_model = self.clf.state_dict()
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print('Early stopping triggered')
                    break

            print('Epoch:', epoch, 
                  'Train Loss:', total_loss/total, 'Train Accuracy:', correct/total, "Train AUC:", print_AUC(-torch.cat(all_probs)[torch.cat(all_labels)==1],-torch.cat(all_probs)[torch.cat(all_labels)==0]),
                  'Val Loss', val_loss/val_total, 'Val Accuracy:', val_correct/val_total, "Val AUC:", print_AUC(-torch.cat(all_probs_val)[torch.cat(all_labels_val)==1],-torch.cat(all_probs_val)[torch.cat(all_labels_val)==0]))
        
        self.clf.load_state_dict(best_model)

        os.makedirs(os.path.dirname(self.clf_name), exist_ok=True)
        torch.save((self.clf,self.feature_set),self.clf_name if self.clf_name.endswith(".pt") else self.clf_name+".pt")

        return -torch.cat(all_probs), torch.cat(all_labels), -torch.cat(all_probs_val), torch.cat(all_labels_val)

    def compute_statistic(self, dataset, batch_size, num_samples=None, device=None):
        """
        Compute the neural network statistic for a given dataloader.

        Args:
            dataset (Dataset): input dataset to compute statistic over
            batch_size (int): batch size to evaluate over
            num_samples (Optional[int]): number of samples to compute over.
                If None, then comptues over whole dataset
            device (Optional[str]): e.g. "cuda"
        Returns:
            torch.Tensor or list: loss of input IDs
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.clf is None:
            raise Exception("Please call .train_model() to train the classifier first.")
        
        tensor_dataset = TensorDataset(dataset)
        dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False)
        all_probs = []
        for step, data in enumerate(dataloader):
            if step>math.ceil(num_samples/batch_size):
                break
            probs = self.clf(data[0].to(device).float())[:,0]
            all_probs.append(probs.detach().cpu())
        return -torch.cat(all_probs)[:num_samples]


class NeuralNetwork(nn.Module):
    def __init__(self,input_size,clf_size):
        super().__init__()
        if clf_size=="small":
            self.layers = nn.Sequential(
                torch.nn.Linear(input_size, 250),
                torch.nn.ReLU(),
                torch.nn.Linear(250,100),
                torch.nn.ReLU(),
                torch.nn.Linear(100,10),
                torch.nn.ReLU(),
                torch.nn.Linear(10,1),
                torch.nn.Sigmoid()
            )
        elif clf_size=="big":
            self.layers = nn.Sequential(
                torch.nn.Linear(input_size, 500),
                torch.nn.ReLU(),
                torch.nn.Linear(500,250),
                torch.nn.ReLU(),
                torch.nn.Linear(250,100),
                torch.nn.ReLU(),
                torch.nn.Linear(100,10),
                torch.nn.ReLU(),
                torch.nn.Linear(10,1),
                torch.nn.Sigmoid()
            )
        else:
            raise NotImplementedError(f"Model architecture '{clf_size}' not recognized.")

    def forward(self, x):
        return self.layers(x)