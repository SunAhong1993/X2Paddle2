from .utils import *

class ClassConv2d(object):
    def __init__(self, func_name, pytorch_api_name, *args, **kwargs):
        self.func_name = func_name
        self.pytorch_api_name = pytorch_api_name
        self.args = args
        self.kwargs = kwargs  
        
    def name_process(self):
        pass
    
    def value_process(self):
        pass
    
    def check(self):
        pass
    
    def run(self):
        new_kwargs = api_args2kwargs(self.pytorch_api_name, self.args)
        self.name_process()
        self.value_process()
        self.value_check()
        return generate_api_code()


class ClassLoss(object):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        self.func_name = func_name
        self.pytorch_api_name = pytorch_api_name
        self.args = args
        self.kwargs = kwargs  
        
    def process_attrs(self):
        pass
    
    def delete_attrs(self):
        delete_key(self.kwargs, "size_average")
        delete_key(self.kwargs, "reduce")
    
    def check_attrs(self):
        pass
    
    def run(self):
        same_attr_count = 1
        if len(self.args) > same_attr_count:
            new_kwargs = api_args2kwargs(self.pytorch_api_name, self.args[same_attr_count:])
            self.kwargs.update(new_kwargs)
        self.check_attrs()
        self.process_attrs()
        self.delete_attrs()
        return [], generate_api_code(self.func_name, self.args, self.kwargs), []
    
class ClassCrossEntropyLoss(ClassLoss):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super(ClassCrossEntropyLoss, self).__init__(func_name, pytorch_api_name, args, kwargs, target_name)
        
class ClassBCEWithLogitsLoss(ClassLoss):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        super(ClassBCEWithLogitsLoss, self).__init__(func_name, pytorch_api_name, args, kwargs, target_name)