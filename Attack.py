from attack_utils import *

class MIA:
    def __init__(self, model_path, model_revision=None, cache_dir=None):
        """
        Base class for all membership inference attacks. Contains a "base" model image. 
            model_path: path to the model to be attacked
            model_revision: revision of the model to be attacked
            cache_dir: directory to cache the model
        """
        self.model_path      = model_path
        self.model_name      = self.model_path.split("/")[-1]
        self.model_revision  = model_revision
        self.cache_dir       = cache_dir
    
    def get_statistics(self):
        raise NotImplementedError()

    def get_default_title(self):
        raise NotImplementedError()

    def attack_plot_ROC(self, title=None, log_scale=False, show_plot=True, save_name=None):
        """
        Plot ROC curve for the attack
        """
        train_statistics, val_statistics = self.get_statistics()
        if title is None:
            title = self.get_default_title()
        if save_name is None:
            save_name = self.get_default_title() + (" log.png" if log_scale else ".png")
        plot_ROC(train_statistics, val_statistics, title, log_scale=log_scale, show_plot=show_plot, save_name=save_name)