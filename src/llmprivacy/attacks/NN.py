import os
import math
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from .Attack import MIA
from ..utils.attack_utils import *
from ..utils.plot_utils import *
from sklearn.preprocessing import StandardScaler

class NN(MIA):
    def __init__(self, clf_name, feature_set, clf_size, model_name, model_revision=None, model_cache_dir=None):
        self.clf_name = clf_name
        self.clf = None
        self.clf_size = None
        self.scaler = None
        self.feature_set = feature_set
        self.model_name = model_name
        self.model_revision = model_revision
        self.model_cache_dir = model_cache_dir
    
    @classmethod
    def get_default_name(cls, clf_name, model_name, model_revision, seed, tag):
        """
        Generates a default experiment name. Also ensures its validity with makedirs.

        Args:
            model_name (str): Huggingface model name
            model_revision (str): model revision name
            num_samples (int): number of training samples
            seed (int): random seed
        Returns:
            string: informative name of experiment
        """
        os.makedirs("results/NN", exist_ok=True)
        return f"results/NN/NN_{clf_name.replace('/','-')}_{model_name.replace('/','-')}_{model_revision.replace('/','-')}_seed={seed}_tag={tag}"

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

    def train_clf(self, features, labels, clf_size, epochs, batch_size, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Take train and validation data and train neural network as MIA.

        Args:
            features (torch.Tensor): Features for training supervised MIA 
            labels (torch.Tensor): Labels for train data (binary, 1 is train)
            clf_size (str): which neural net architecture to use
            epochs (int): number of epochs to train
            batch_size (int): batch size to train with
            device (str): device to train on
                    
        Returns:
            tuple[torch.Tensor]: train predictions 
        """

        self.clf = NeuralNetwork(features.shape[1],clf_size).to(device)

        dataset = TensorDataset(features, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = Adam(self.clf.parameters())
        loss_fn = nn.BCELoss()

        # Training
        for epoch in tqdm(range(epochs)):
            self.clf.train()
            total_loss = 0
            correct = 0
            total = 0
            all_probs = []
            all_labels = []

            for step, (data, target) in enumerate(dataloader):
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

            print('Epoch:', epoch, 'Loss:', total_loss/total, 'Accuracy:', correct/total, "AUC:", print_AUC(-torch.cat(all_probs)[torch.cat(all_labels)==1],-torch.cat(all_probs)[torch.cat(all_labels)==0]))

        os.makedirs(os.path.dirname(self.clf_name), exist_ok=True)
        torch.save((self.clf,self.feature_set),self.clf_name if self.clf_name.endswith(".pt") else self.clf_name+".pt")

        return -torch.cat(all_probs), torch.cat(all_labels)

    def compute_statistic(self, dataset, batch_size, num_samples=None, device="cuda" if torch.cuda.is_available() else "cpu"):
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