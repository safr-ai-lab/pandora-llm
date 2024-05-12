import os
import torch
from .Attack import MIA
from ..utils.attack_utils import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class LogReg(MIA):
    def __init__(self, clf_name, feature_set, model_name, model_revision=None, model_cache_dir=None):
        self.clf_name = clf_name
        self.clf = None
        self.scaler = None
        self.feature_set = feature_set
        self.model_name = model_name
        self.model_revision = model_revision
        self.model_cache_dir = model_cache_dir
    
    @classmethod
    def get_default_name(cls, clf_name, model_name, model_revision, seed,tag):
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
        os.makedirs("results/LogReg", exist_ok=True)
        return f"results/LogReg/LogReg_{clf_name.replace('/','-')}_{model_name.replace('/','-')}_{model_revision.replace('/','-')}_seed={seed}_tag={tag}"

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
        processed_features = self.scaler.transform(processed_features)
        
        # Return
        if labels is not None:
            labels = labels[mask_out_infinite]
            return processed_features, labels
        else:
            return processed_features

    def train_clf(self, features, labels, max_iter, seed=229):
        """
        Take train and validation data and train logistic regression as MIA.

        Args:
            features (torch.Tensor): Features for training supervised MIA 
            labels (torch.Tensor): Labels for train data (binary, 1 is train)
            max_iter (int): number of iterations of logistic regression
            seed (int): Seed of logistic regression
                    
        Returns:
            tuple[torch.Tensor]: train predictions 
        """

        self.clf = LogisticRegression(max_iter=max_iter,random_state=seed).fit(features, labels)

        os.makedirs(os.path.dirname(self.clf_name), exist_ok=True)
        torch.save((self.clf,self.feature_set),self.clf_name if self.clf_name.endswith(".pt") else self.clf_name+".pt")

        return -self.clf.predict_proba(features)[:,1]

    def compute_statistic(self, dataset, num_samples=None):
        """
        Compute the LR statistic for a given dataloader. Not implemented. 

        Args:
            dataset (Dataset): input dataset to compute statistic over
            num_samples (Optional[int]): number of samples to compute over.
                If None, then comptues over whole dataset
            device (Optional[str]): e.g. "cuda"
            model_half (Optional[bool]): whether to use model_half
            accelerator (Optional[Accelerator]): accelerator object
        Returns:
            torch.Tensor or list: loss of input IDs
        """
        if self.clf is None:
            raise Exception("Please call .train_model() to train the classifier first.")
        
        return -self.clf.predict_proba(dataset[:num_samples] if num_samples is not None else dataset)[:,1]