from .utils import *

class FuncSave(object):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        self.func_name = func_name
        self.pytorch_api_name = pytorch_api_name
        self.args = args
        self.kwargs = kwargs  
        
    def process_attrs(self):
        pass
    
    def delete_attrs(self):
        delete_key(self.kwargs, "pickle_module")
    
    def check_attrs(self):
        pass
    
    def run(self):
        same_attr_count = 2
        if len(self.args) > same_attr_count:
            new_kwargs = api_args2kwargs(self.pytorch_api_name, self.args[same_attr_count:])
            self.kwargs.update(new_kwargs)
        self.check_attrs()
        self.process_attrs()
        self.delete_attrs()
        return [], generate_api_code(self.func_name, self.args, self.kwargs), []
    

class FuncLoad(object):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        self.func_name = func_name
        self.pytorch_api_name = pytorch_api_name
        self.args = args
        self.kwargs = kwargs  
        
    def process_attrs(self):
        pass
    
    def delete_attrs(self):
        delete_key(self.kwargs, "pickle_module")
        delete_key(self.kwargs, "map_location")
    
    def check_attrs(self):
        pass
    
    def run(self):
        same_attr_count = 2
        if len(self.args) > same_attr_count:
            new_kwargs = api_args2kwargs(self.pytorch_api_name, self.args[same_attr_count:])
            self.kwargs.update(new_kwargs)
        self.check_attrs()
        self.process_attrs()
        self.delete_attrs()
        return [], generate_api_code(self.func_name, self.args, self.kwargs), []
        

class FuncSetDevice(object):
    def __init__(self, func_name, pytorch_api_name, args, kwargs, target_name=None):
        self.func_name = func_name
        self.pytorch_api_name = pytorch_api_name
        self.args = args
        self.kwargs = kwargs  
        self.target_name = target_name
        self.useful_attrs = dict()
        
    def process_attrs(self):
        self.useful_attrs["device"] = self.args[0]
        self.args[0] = self.target_name
    
    def delete_attrs(self):
        pass
    
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
        insert_codes = list()
        insert_codes.append("{} = {}".format(self.target_name, self.useful_attrs["device"]))
        insert_codes.append("{} = {}.replace('cuda', 'gpu')".format(self.target_name, self.target_name))  
        print("000000000", insert_codes)
        return insert_codes, generate_api_code(self.func_name, self.args, self.kwargs), []