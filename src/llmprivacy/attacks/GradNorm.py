import os
from transformers import AutoModelForCausalLM
from .Attack import MIA
from ..utils.attack_utils import *
from ..utils.plot_utils import *

class GradNorm(MIA):
    """
    GradNorm thresholding attack
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = None

    def load_model(self):
        """
        Loads model into memory
        """
        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, revision=self.model_revision, cache_dir=self.model_cache_dir)
        else:
            raise Exception("Model has already been loaded; please call .unload_model() first!")

    def unload_model(self):
        """
        Unloads model from memory
        """
        self.model = None

    def compute_gradients(self, dataloader, norms=None, num_batches=None, device=None, model_half=None, accelerator=None, max_length=None):
        """
        Compute the gradient norms for a given dataloader, using the specified norms.

        Args:
            dataloader (DataLoader): input data to compute statistic over
            num_batches (Optional[int]): number of batches of the dataloader to compute over.
                If None, then comptues over whole dataloader
            norms (Optional[list[int,float]]): list of norm orders
            device (Optional[str]): e.g. "cuda"
            model_half (Optional[bool]): whether to use model_half
            accelerator (Optional[Accelerator]): accelerator object
        Returns:
            torch.Tensor or list: grad norm of input IDs
        """
        if self.model is None:
            raise Exception("Please call .load_model() to load the model first.")
        if accelerator is not None:
            self.model.gradient_checkpointing_enable() # TODO
            self.model, dataloader, = accelerator.prepare(self.model, dataloader)
            if accelerator.is_main_process:
                subprocess.call(["python", "model_embedding.py",
                    "--model_name", self.model_name,
                    "--model_revision", self.model_revision,
                    "--model_cache_dir", self.model_cache_dir,
                    "--save_path", "results/GRAD/embedding.pt",
                    "--model_half" if model_half else ""
                    ]
                )
            accelerator.wait_for_everyone()
            embedding_layer = torch.load("results/GRAD/embedding.pt")
            self.model.train()
        else:
            embedding_layer = self.model.get_input_embeddings().weight
        if norms is None:
            norms = [1,2,float("inf")]
        return compute_dataloader_all_norms(model=self.model,embedding_layer=embedding_layer,norms=norms,dataloader=dataloader,num_batches=num_batches,device=device,model_half=model_half)

    @classmethod
    def get_default_name(cls, model_name, model_revision, num_samples, start_index, seed, datasubset):
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
        os.makedirs("results/GradNorm", exist_ok=True)
        if datasubset is None:
            return f"results/GradNorm/GradNorm_{model_name.replace('/','-')}_{model_revision.replace('/','-')}_N={num_samples}_S={start_index}_seed={seed}"
        else:
            return f"results/GradNorm/GradNorm_{model_name.replace('/','-')}_{model_revision.replace('/','-')}_N={num_samples}_S={start_index}_seed={seed}_datasubset={datasubset}"


    def attack_plot_ROC(self, train_gradients, val_gradients, title, log_scale=False, show_plot=False, save_name=None):
        """
        Generates and displays or saves a plot of the ROC curve for the membership inference attack.

        For GradNorm, this method plots a series of ROC curves for differing values of norm order and input vs. model gradient.

        This method uses the inputted statistics to create a ROC curve that 
        illustrates the performance of the attack. The plot can be displayed in a log scale, 
        shown directly, or saved to a file.

        Args:
            train_statistics (dict[str,float]]): Gradients of the training set. Lower means more like train.
            val_statistics (dict[str,float]]): Gradients of the training set. Lower means more like train.
            title (str): The title for the ROC plot.
            log_scale (bool, optional): Whether to plot the ROC curve on a logarithmic scale. 
                Defaults to False.
            show_plot (bool, optional): Whether to display the plot. If False, the plot is not 
                shown but is saved directly to the file specified by `save_name`. Defaults to True.
            save_name (str, optional): The file name or path to save the plot image. If not 
                specified, the default name is generated by the given title with an 
                appropriate file extension. Defaults to None.
        """
        if save_name is None:
            save_name = title + (" log.png" if log_scale else ".png")
        
        train_statistics = []
        val_statistics = []
        labels = []
        for grad_type in train_gradients.keys():
            train_stats = train_gradients[grad_type]
            val_stats = val_gradients[grad_type]
            train_statistics.append(train_stats)
            val_statistics.append(val_stats)
            labels.append(grad_type)
        plot_ROC_multiple(train_statistics, val_statistics, title, labels, log_scale=log_scale, show_plot=show_plot, save_name=save_name)