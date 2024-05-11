from .Attack import MIA
from ..utils.attack_utils import *
from transformers import AutoModelForCausalLM
from torch.utils.data import DataLoader
import torch
import subprocess
import os
from typing import Optional

class TensorToDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, requires_grad=False)
        self.y = torch.tensor(y, requires_grad=False)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class NN(MIA):
    def __init__(self, model_name, ft_model_name, model_revision=None, model_cache_dir=None, ft_model_revision=None, ft_model_cache_dir=None):
        self.model_name         = model_name
        self.model_revision     = model_revision
        self.model_cache_dir    = model_cache_dir
        self.nn_model = None 
    
    @classmethod
    def get_default_name(cls, model_name, model_revision, jl_data, test_frac, only_x, only_theta, seed):
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
        return f"results/NN/NN_{model_name.replace('/','-')}_{model_revision.replace('/','-')}_jl_data={jl_data}_test_frac={test_frac}_onlyx={only_x}_onlytheta={only_theta}_seed={seed}"

    def train_model(self, Xtrain: np.array, Xtest: np.array, ytrain: np.array, ytest: np.array, large_model: bool, nn_epochs: int, nn_savename: str, data_savename: str, seed=229):
        """
        Take train and validation data and train logistic regression as MIA.

        Args:
            Xtrain (np.array): Train data for supervised MIA 
            Xtest (np.array): Test data for supervised MIA
            ytrain (np.array): Labels for train data (0 or 1)
            ytest (np.array): Labels for test data (0 or 1)
            large_model (bool): Use big model (Y/N)
            nn_savename (str): save location of NN
            data_savename (str): save location of NN predictions
            seed (int): Seed of logistic regression
                    
        Returns:
            tuple[torch.Tensor, torch.Tensor]: scores of train and validation data
        """
        
        ## Neural network-related code
        train_dataset = TensorToDataset(Xtrain, ytrain)
        test_dataset = TensorToDataset(Xtest, ytest)

        # Devicing
        train_dataset.X.to(device)
        train_dataset.y.to(device)
        test_dataset.X.to(device)
        test_dataset.y.to(device)

        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=True)
        if large_model:
            model = torch.nn.Sequential(
                                        torch.nn.Linear(all_train.shape[1], 500),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(500,250),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(250,100),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(100,10),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(10,1),
                                        torch.nn.Sigmoid()
                                    ).to(device)
        else:
            model = torch.nn.Sequential(
                                        torch.nn.Linear(all_train.shape[1], 250),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(250,100),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(100,10),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(10,1),
                                        torch.nn.Sigmoid()
                                    ).to(device)

        optimizer = torch.optim.Adam(model.parameters())
        loss_fn = torch.nn.BCELoss()
        for epoch in tqdm(range(nn_epochs)):
            model.train()
            total_loss = 0
            nbatches = 0

        ## Train
        for step, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data.to(device).float())
            loss = loss_fn(output[:,0], target.to(device))
            total_loss += loss.item()
            nbatches += 1
            loss.backward()
            optimizer.step()

        ## Test
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for _, (data, target) in enumerate(test_loader):
                output = model(data.to(device).float())
                pred = (output > 0.5).int().cpu()
                correct += (pred[:,0] == target).sum().item()
                total += target.size(0)
            newdata_ypred = model(torch.tensor(Xtest).to(device).float())[:,0].cpu().detach().numpy()
            print('Epoch:', epoch, 'Loss:', total_loss/nbatches, 'Accuracy:', correct / total, "AUC:", print_AUC(-newdata_ypred[newdata_ytest==1],-newdata_ypred[newdata_ytest==0]))

        newdata_ypred = model(torch.tensor(Xtest).to(device).float())[:,0].cpu().detach().numpy()
        torch.save(model,nn_savename)
        
        # Save data 
        self.train_stats, self.val_stats = torch.tensor(newdata_ypred[ytest==1]), torch.tensor(newdata_ypred[ytest==0])
        torch.save((self.train_stats, self.val_stats), data_savename)
        
        return  self.train_stats, self.val_stats


    def compute_statistic(self, dataloader, num_batches=None, device=None, model_half=None, accelerator=None):
        """
        Compute the LR statistic for a given dataloader. Not implemented. 

        Args:
            dataloader (DataLoader): input data to compute statistic over
            num_batches (Optional[int]): number of batches of the dataloader to compute over.
                If None, then comptues over whole dataloader
            device (Optional[str]): e.g. "cuda"
            model_half (Optional[bool]): whether to use model_half
            accelerator (Optional[Accelerator]): accelerator object
        Returns:
            torch.Tensor or list: loss of input IDs
        """
        if self.nn_model is None:
            raise Exception("Please call .train_model() to train the LR first.")
        if accelerator is not None:
            self.model, dataloader, = accelerator.prepare(self.model, dataloader)
        
        return # TODO - currently, no need to evaluate on a bespoke dataloader because data prepared internally in diff format