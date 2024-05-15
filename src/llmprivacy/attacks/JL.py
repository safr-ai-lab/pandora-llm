import os
import math
from transformers import AutoModelForCausalLM
from .Attack import MIA
from ..utils.attack_utils import *
from ..utils.plot_utils import *
from trak.projectors import CudaProjector, ProjectionType, ChunkedCudaProjector, BasicProjector

class JL(MIA):
    """
    JL features attack
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = None

    @classmethod
    def get_default_name(cls, model_name, model_revision, num_samples, start_index, seed):
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
        os.makedirs("results/JL", exist_ok=True)
        return f"results/JL/JL_{model_name.replace('/','-')}_{model_revision.replace('/','-')}_N={num_samples}_S={start_index}_seed={seed}"
        
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

    def compute_jl(self, 
        dataloader,
        proj_x_to,
        proj_each_layer_to,
        proj_type,
        proj_seed=229,
        num_batches=None,
        device=None, model_half=None, accelerator=None, max_length=None
    ):
        """
        Compute the JL projection of the gradients for a given dataloader.

        Args:
            dataloader (DataLoader): input data to compute statistic over
            proj_x_to (int): the number of dimensions to project embedding gradient to
            proj_each_layer_to (int): the number of dimensions to project each layer to
            proj_type (str): the projection type (either 'normal' or 'rademacher')
            proj_seed (int): the random seed to use in the projection
            num_batches (Optional[int]): number of batches of the dataloader to compute over.
                If None, then comptues over whole dataloader
            device (Optional[str]): e.g. "cuda"
            model_half (Optional[bool]): whether to use model_half
            accelerator (Optional[Accelerator]): accelerator object
        Returns:
            torch.Tensor or list: grad norm of input IDs
        """
        if self.model is None:
            raise Exception("Please call .load_model() to load the model first.")
        print(f"In total, the number of features will be: {sum(1 for _ in self.model.named_parameters()) * proj_each_layer_to}.")

        # Retrieve embedding layer
        if accelerator is not None:
            self.model.gradient_checkpointing_enable() # TODO
            self.model, dataloader, = accelerator.prepare(self.model, dataloader)
            if accelerator.is_main_process:
                subprocess.call(["python", "model_embedding.py",
                    "--model_name", self.model_name,
                    "--model_revision", self.model_revision,
                    "--model_cache_dir", self.model_cache_dir,
                    "--save_path", "results/JL/embedding.pt",
                    "--model_half" if model_half else ""
                    ]
                )
            accelerator.wait_for_everyone()
            embedding_layer = torch.load("results/JL/embedding.pt")
            self.model.train()
        else:
            embedding_layer = self.model.get_input_embeddings().weight
        
        # Project each type of data with a JL dimensionality reduction
        projectors = {}                 
        projectors["x"] = BasicProjector(
            grad_dim=next(self.model.parameters()).shape[1]*2048,
            proj_dim=proj_x_to,
            seed=proj_seed,
            proj_type=ProjectionType(proj_type),
            device=device,
            block_size=1
        ) #TODO replace with CudaProjector
        
        for i, (name,param) in enumerate(self.model.named_parameters()):
            projectors[(i,name)] = BasicProjector(
                grad_dim=math.prod(param.size()),
                proj_dim=proj_each_layer_to,
                seed=proj_seed,
                proj_type=ProjectionType(proj_type),
                device=device,
                block_size=1
            )
    
        return compute_dataloader_jl(model=self.model,embedding_layer=embedding_layer,projector=projectors,dataloader=dataloader,nbatches=num_batches,device=device,half=model_half).cpu() 
        
    def compute_jl_balanced(self, 
        dataloader,
        proj_x_to,
        proj_group_to,
        proj_type,
        proj_seed=229,
        num_splits=8,
        num_batches=None,
        device=None, model_half=None, accelerator=None, max_length=None
    ):
        """
        Compute the JL projection of the gradients for a given dataloader, more equally distributed across num_split splits

        Args:
            dataloader (DataLoader): input data to compute statistic over
            proj_x_to (int): the number of dimensions to project embedding gradient to
            proj_group_to (int): the number of dimensions to project each group to
            proj_type (str): the projection type (either 'normal' or 'rademacher')
            proj_seed (int): the random seed to use in the projection
            num_splits (int): how many splits of parameters to compute JL over
            num_batches (Optional[int]): number of batches of the dataloader to compute over.
                If None, then comptues over whole dataloader
            device (Optional[str]): e.g. "cuda"
            model_half (Optional[bool]): whether to use model_half
            accelerator (Optional[Accelerator]): accelerator object
        Returns:
            torch.Tensor or list: grad norm of input IDs
        """
        if self.model is None:
            raise Exception("Please call .load_model() to load the model first.")
        print(f"In total, the number of features will be: {num_splits * proj_group_to}.")

        # Retrieve embedding layer
        if accelerator is not None:
            self.model.gradient_checkpointing_enable() # TODO
            self.model, dataloader, = accelerator.prepare(self.model, dataloader)
            if accelerator.is_main_process:
                subprocess.call(["python", "model_embedding.py",
                    "--model_name", self.model_name,
                    "--model_revision", self.model_revision,
                    "--model_cache_dir", self.model_cache_dir,
                    "--save_path", "results/JL/embedding.pt",
                    "--model_half" if model_half else ""
                    ]
                )
            accelerator.wait_for_everyone()
            embedding_layer = torch.load("results/JL/embedding.pt")
            self.model.train()
        else:
            embedding_layer = self.model.get_input_embeddings().weight
        
        ## JL balanced Features
        sizes = []
        for i, (name,param) in enumerate(self.model.named_parameters()):
            sizes.append(math.prod(param.size()))
        
        def balanced_partition(sizes, num_groups):
            # Pair each size with its original index
            sizes_with_indices = list(enumerate(sizes))
            # Sort sizes in descending order while keeping track of indices
            sizes_with_indices.sort(key=lambda x: x[1], reverse=True)
            
            # Initialize groups and their sums
            groups = [[] for _ in range(num_groups)]
            group_sums = [0] * num_groups
            group_indices = [[] for _ in range(num_groups)]
            
            # Assign each size to the group with the smallest current sum
            for index, size in sizes_with_indices:
                min_index = group_sums.index(min(group_sums))
                groups[min_index].append(size)
                group_indices[min_index].append(index)
                group_sums[min_index] += size

            return groups, group_sums, group_indices
        groups, sums, indices = balanced_partition(sizes, num_splits)
        print(f"Split groups: {groups}")
        print(f"Split sums: {sums}")
        print(f"Split group indices: {indices}")

        projectors = {}
        for i in range(num_splits):
            projectors[i] = CudaProjector(
                grad_dim=sums[i], 
                proj_dim=proj_group_to,
                seed=proj_seed,
                proj_type=ProjectionType(proj_type),
                device='cuda',
                max_batch_size=32,
            )
        
        projectors["x"] = BasicProjector(
            grad_dim=next(self.model.parameters()).shape[1]*2048,
            proj_dim=proj_x_to,
            seed=proj_seed,
            proj_type=ProjectionType(proj_type),
            device='cuda',
            block_size=1,
        )
        return compute_dataloader_jl_balanced(model=self.model, embedding_layer=embedding_layer, dataloader=dataloader, projector=projectors, indices=indices, device=device, nbatches=num_batches, half=model_half).cpu() 
        
    def compute_jl_model_stealing(self, 
        dataloader,
        svd_dataloader,
        proj_type,
        proj_dim=512,
        proj_seed=229,
        num_batches=None,
        device=None, model_half=None, accelerator=None, max_length=None
    ):
        """
        Compute the JL projection of the gray-box model-stealing attack for a given dataloader.

        Args:
            dataloader (DataLoader): input data to compute statistic over
            svd_dataloader (DataLoader): svd dataloader
            proj_type (str): the projection type (either 'normal' or 'rademacher')
            proj_dim (int): the number of dimensions to project to
            proj_seed (int): the random seed to use in the projection
            num_batches (Optional[int]): number of batches of the dataloader to compute over.
                If None, then comptues over whole dataloader
            device (Optional[str]): e.g. "cuda"
            model_half (Optional[bool]): whether to use model_half
            accelerator (Optional[Accelerator]): accelerator object
        Returns:
            torch.Tensor or list: grad norm of input IDs
        """
        if self.model is None:
            raise Exception("Please call .load_model() to load the model first.")
        projectors = {}
        input_size = next(self.model.parameters()).shape[1] * next(self.model.parameters()).shape[0]
        projectors["embed_out"] = CudaProjector(
            grad_dim=input_size, 
            proj_dim=proj_dim, 
            seed=proj_seed,
            proj_type=ProjectionType(proj_type),
            device='cuda',
            max_batch_size=8
        )
        
        # Computing Logits for svd_dataloader
        print("Computing logits for svd_dataloader")
        dataloader_logits = compute_dataloader_logits_embedding(self.model, svd_dataloader, device, half=model_half).T.float().to(device)
        torch.save(dataloader_logits, "results/JL/logits.pt")
        last_layer = [m for m in self.model.parameters()][-1]
        
        # Generate matrix U @ torch.diag(S) which is equal to embedding projection up to symmetries
        U, S, _ = torch.linalg.svd(dataloader_logits,full_matrices=False)
        svd_embedding_projection_layer = U @ torch.diag(S)

        # Identify base change to convert regular gradients to gradients we can access
        print("Computing random basis change")
        base_change = torch.linalg.pinv(last_layer.float()).to('cpu') @ svd_embedding_projection_layer.to('cpu')
        projectors["random_basis_change"] = torch.linalg.inv(base_change).T  # NOT A PROJECTOR, code just puts it in projectors dict for some reason

        print("Computing Carlini Gray Box Features")
        return compute_dataloader_basis_changes(model=self.model, dataloader=dataloader, projector=projectors, device=device, nbatches=num_batches, half=model_half).cpu() 