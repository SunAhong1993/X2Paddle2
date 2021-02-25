from .utils import *

class ClassAdam(object):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        self.func_name = func_name
        self.pytorch_api_name = pytorch_api_name
        self.args = args
        self.kwargs = kwargs  
        
    def process_attrs(self):
        rename_key(self.kwargs, "params", "parameters")
        rename_key(self.kwargs, "lr", "learning_rate")
        rename_key(self.kwargs, "eps", "epsilon")
        if "betas" in self.kwargs:
            betas = self.kwargs.pop("betas")
            self.kwargs["beta1"] = betas[0]
            self.kwargs["beta2"] = betas[1]
        
     
    def delete_attrs(self):
        pass
    
    def check_attrs(self):
        assert "amsgrad" not in self.kwargs or not self.kwargs["amsgrad"], "The amsgrad in torch.optim.Adam must be False!"
    
    def run(self):
        same_attr_count = 0
        if len(self.args) > same_attr_count:
            new_kwargs = api_args2kwargs(self.pytorch_api_name, self.args[same_attr_count:])
            self.kwargs.update(new_kwargs)
            self.args = self.args[:same_attr_count]
        self.check_attrs()
        self.process_attrs()
        self.delete_attrs()
        return [], generate_api_code(self.func_name, self.args, self.kwargs), []

