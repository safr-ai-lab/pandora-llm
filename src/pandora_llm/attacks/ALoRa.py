from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM
from .Attack import MIA

####################################################################################################
# MAIN CLASS
####################################################################################################
class ALoRa(MIA):
    """
    Approximate loss ratio thresholding attack (vs. pre-training)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_statistic(self, dataloader, learning_rate, num_batches=None, device=None, model_half=None, accelerator=None):
        """
        Compute the approximate loss ratio statistic for a given dataloader.
        Assumes batch size of 1.

        Args:
            dataloader (DataLoader): input data to compute statistic over
            learning_rate (float): learning rate
            num_batches (Optional[int]): number of batches of the dataloader to compute over.
                If None, then comptues over whole dataloader
            device (Optional[str]): e.g. "cuda"
            model_half (Optional[bool]): whether to use model_half
            accelerator (Optional[Accelerator]): accelerator object
        Returns:
            tuple[torch.Tensor,torch.Tensor]: loss after the gradient descent step and loss before the gradient descent step
        """
        if self.model is None:
            raise Exception("Please call .load_model() to load the model first.")
        if dataloader.batch_size!=1:
            raise Exception("ALoRa is only implemented for batch size 1")
        
        base_statistics = []
        stepped_statistics = []
        # Setup device
        if accelerator is not None:
            self.model, dataloader, = accelerator.prepare(self.model, dataloader)
        else:
            self.model.to(device)
            if model_half:
                self.model.half()
        for batchno, data_x in tqdm(enumerate(dataloader),total=len(dataloader)):
            if num_batches is not None and batchno >= num_batches:
                break
            # 1. Forward pass
            data_x = data_x["input_ids"]
            if accelerator is None:
                data_x = data_x.to(device)
            outputs = self.model(data_x, labels=data_x)
            initial_loss = outputs.loss
            initial_loss.backward() 
            base_statistics.append(initial_loss.detach().cpu())

            # 2. Perform gradient descent
            with torch.no_grad():
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.add_(param.grad, alpha=-learning_rate)  # Gradient descent step

            # 3. Compute the loss after update
            output_after_ascent = self.model(data_x, labels=data_x)
            new_loss = output_after_ascent.loss
            stepped_statistics.append(new_loss.detach().cpu())

            # 4. Restore model to original state
            # self.unload_model()
            # self.load_model()
            with torch.no_grad():
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.add_(param.grad, alpha=learning_rate)  # Reverse gradient descent step
            self.model.zero_grad()  # Reset gradients
            del data_x, outputs, initial_loss, output_after_ascent, new_loss
            torch.cuda.empty_cache()

        return torch.tensor(stepped_statistics), torch.tensor(base_statistics)