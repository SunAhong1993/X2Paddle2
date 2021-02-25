import ast
import astor
from x2paddle.code_convertor.pytorch.mapper import *
import copy

class ImportInfo:
    PYTORCH_FROM = None
    PADDLE_FROM = None
    PYTORCH_IMPORT = None
    PADDLE_IMPORT = None
    AS = None
    PYTORCH_IMPORT_STR = None
    PADDLE_IMPORT_STR = None
    
    
class Convertor(ast.NodeVisitor):
    def __init__(self, root):
        self.root = root
        self.scope_and_imports = list() # stack 
        self._stack = list()
        
    def _get_scope_node(self):
        scope_node = None
        for i in range(len(self.scope_and_imports)):
            i = - (i + 1)
            import_info = self.scope_and_imports[i]
            if not isinstance(import_info, ImportInfo) and not isinstance(import_info, ast.Assign):
                scope_node = import_info
                break
        return scope_node 
    
    def _get_father_node(self):
        return self._stack[-2]
    
    def _get_current_index(self, scope_node, node):
        current_id = 0
        for i, n in enumerate(scope_node.body):
            if node == n:
                current_id = i
                break
        return current_id
        
    def run(self):
        self.scope_and_imports.append(self.root)
        self.visit(self.root)
        for i, node in enumerate(self.root.body):
            if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                self.root.body.insert(i, ast.parse("from x2paddle import torch2paddle").body[0])
                break
        
    def visit(self, node):
        print("=========", node.__class__.__name__)
#         print(ast.dump(node))
        self._stack.append(node)
        out = super(Convertor, self).visit(node)
        self._stack.pop()
        if out is not None:
            return out
        else:
            try:
                return astor.to_source(node)
            except Exception:
                return None
        
    def visit_ImportFrom(self, node):
        if not node.module.startswith("torch"):
            return
        scope_node = self._get_scope_node()
        current_id = self._get_current_index(scope_node, node)
        scope_node.body.pop(current_id)
        self._from_name = node.module
        son_nodes = node.names
        for son_node in son_nodes:
            copy_node = copy.deepcopy(node)
            copy_node.names = [son_node]
            self.visit_alias(son_node, copy_node)
            scope_node.body.insert(current_id, copy_node)
        self._from_name = None
        
    def visit_Import(self, node):
        self._from_name = None
        son_nodes = getattr(node, "names")
        for son_node in son_nodes:
            self.visit_alias(son_node, node)
            
        
    def visit_alias(self, node, father_node=None):
        import_info = ImportInfo()
        import_info.PYTORCH_FROM = self._from_name
        import_info.PYTORCH_IMPORT = getattr(node, "name", None)
        import_info.AS = getattr(node, "asname", None)
        import_str_list = list()
        if import_info.PYTORCH_FROM is not None:
            import_str_list.append(import_info.PYTORCH_FROM)
        if import_info.PYTORCH_IMPORT is not None:
            import_str_list.append(import_info.PYTORCH_IMPORT)
        import_info.PYTORCH_IMPORT_STR = ".".join(import_str_list)
        if "torch" in import_info.PYTORCH_IMPORT_STR:
            import_info.PADDLE_IMPORT_STR = API_MAPPER[import_info.PYTORCH_IMPORT_STR][0]
            if import_info.PYTORCH_FROM is not None:
                seg = import_info.PADDLE_IMPORT_STR.split(".")
                setattr(node, "name", seg[-1])
                setattr(father_node, "module", import_info.PADDLE_IMPORT_STR.replace("." + seg[-1], ""))
                import_info.PADDLE_IMPORT = seg[-1]
                import_info.PADDLE_FROM = import_info.PADDLE_IMPORT_STR.replace("." + seg[-1], "")
            else:
                setattr(node, "name", import_info.PADDLE_IMPORT_STR) 
                import_info.PADDLE_IMPORT = import_info.PADDLE_IMPORT_STR
            if import_info.AS is None:
                if import_info.PYTORCH_IMPORT != node.name:
                    setattr(node, "asname", import_info.PYTORCH_IMPORT)
        else:
            import_info.PADDLE_IMPORT_STR = import_info.PYTORCH_IMPORT_STR   
        self.scope_and_imports.append(import_info)
        
    def visit_Name(self, node):
        return getattr(node, "id")
    
    def visit_Attribute(self, node):
        value_node = node.value
        attr = node.attr
        name = self.visit(value_node)
        attr_str = name + "." + attr
        if attr_str in API_MAPPER:
            paddle_attr_str = API_MAPPER[attr_str][0]
            father_node = self._get_father_node()
            if not isinstance(father_node, ast.Call):
                # 对torch.float32的处理
                for k, v in father_node.__dict__.items():
                    if v == node:
                        setattr(father_node, k, ast.parse(paddle_attr_str).body[0].value)
                        break
                return paddle_attr_str
        return attr_str
    
    def visit_Num(self, node):
        return getattr(node, "n")
    
    def visit_keyword(self, node):
        key = getattr(node, "arg")
        value_node = getattr(node, "value")
        value = self.visit(value_node)
        return {key: value}
    
    def visit_Tuple(self, node):
        elts_nodes = getattr(node, "elts")
        elts = list()
        for elts_node in elts_nodes:
            elts.append(self.visit(elts_node))
        elts = tuple(elts)
        return elts
    
    def visit_Assign(self, node):
        self.scope_and_imports.append(node)
        self.generic_visit(node)
        
    def visit_Call(self, node):
        # 获取函数名
        func_node = node.func
        func_name = self.visit(func_node) 
        pytorch_api = None
        for i in range(len(self.scope_and_imports)):
            i = - (i + 1)
            import_info = self.scope_and_imports[i]
            if isinstance(import_info, ImportInfo):
                if import_info.AS is not None and func_name.startswith(import_info.AS):
                    if (import_info.PYTORCH_FROM is not None and "torch" in import_info.PYTORCH_FROM) or \
                            (import_info.PYTORCH_IMPORT is not None and "torch" in import_info.PYTORCH_IMPORT):
                        pytorch_api = func_name.replace(import_info.AS, import_info.PYTORCH_IMPORT_STR)
                        break
                elif func_name.startswith(import_info.PYTORCH_IMPORT):
                    if (import_info.PYTORCH_FROM is not None and "torch" in import_info.PYTORCH_FROM) or \
                            (import_info.PYTORCH_IMPORT is not None and "torch" in import_info.PYTORCH_IMPORT):
                        pytorch_api = func_name.replace(import_info.PYTORCH_IMPORT, import_info.PYTORCH_IMPORT_STR)
                        break  
            elif isinstance(import_info, ast.Assign):
                is_customized = False
                for s in astor.to_source(import_info.targets[0]).split(","):
                    if func_name.split(".")[0] == s.strip():
                        is_customized = True
                        break
                if is_customized:
                    break
        if pytorch_api is None:
            self.generic_visit(node)
            return
        paddle_api = API_MAPPER[pytorch_api][0]
        pytorch_api_seg = pytorch_api.split(import_info.PYTORCH_IMPORT)
        if paddle_api.startswith(import_info.PADDLE_IMPORT + ".") or \
                paddle_api.endswith("." + import_info.PADDLE_IMPORT) or  \
                "." + import_info.PADDLE_IMPORT + "." in paddle_api:
            # 此时import的库包含改op (反例：import paddle.nn as nn; nn.utils.utils.clip_grad_value_)
            paddle_api_seg = paddle_api.split(import_info.PADDLE_IMPORT)
            if import_info.AS is None:
                func_name = func_name.replace(import_info.PYTORCH_IMPORT + pytorch_api_seg[-1], 
                                              import_info.PADDLE_IMPORT +paddle_api_seg[-1])
            else:
                func_name = func_name.replace(pytorch_api_seg[-1], paddle_api_seg[-1])
            setattr(node, "func", ast.parse(func_name).body[0].value)
        else:
            setattr(node, "func", ast.parse("torch2paddle." + paddle_api.split("torch2paddle.")[-1]).body[0].value)
            
        
        scope_node = self._get_scope_node()
        
        # 获取args
        args_nodes = getattr(node, "args")
        args_list = list()
        for args_node in args_nodes:
            args_list.append(self.visit(args_node))
                
        # 获取keywords
        keywords_nodes = getattr(node, "keywords")
        kw_dict = dict()
        for keywords_node in keywords_nodes:
            kw_dict.update(self.visit(keywords_node))
            
        if API_MAPPER[pytorch_api][1] is None:
            return
                
        target_name = None
        father_node = self._get_father_node()
        if father_node.__class__.__name__ == "Assign":
            target_node = father_node.targets[0]
            target_name = self.visit(target_node)
        mapper = API_MAPPER[pytorch_api][1](func_name, pytorch_api, args_list, kw_dict, target_name)
        prefix_insert_codes, new_code, suffix_insert_codes = mapper.run()
        new_call_node = ast.parse(new_code).body[0].value
        setattr(node, "args", new_call_node.args)
        setattr(node, "keywords", new_call_node.keywords)
        for i, n in enumerate(scope_node.body):
            if father_node == n:
                for code in prefix_insert_codes:
                    scope_node.body.insert(i, ast.parse(code).body[0])
                    i += 1 
                break
        for i, n in enumerate(scope_node.body):
            if father_node == n:
                j = i + 1
                for code in suffix_insert_codes:
                    scope_node.body.insert(j, ast.parse(code).body[0])
                    j += 1  
                break
        
    def visit_FunctionDef(self, node):
        self.scope_and_imports.append(node)
        self.generic_visit(node)
        last_node = self.scope_and_imports.pop(-1)
        while isinstance(last_node, ImportInfo): 
            last_node = self.scope_and_imports.pop(-1)
        
        
    def visit_ClassDef(self, node):
        self.scope_and_imports.append(node)
        self.generic_visit(node)
        last_node = self.scope_and_imports.pop(-1)
        while isinstance(last_node, ImportInfo): 
            last_node = self.scope_and_imports.pop(-1)
            
    def visit_If(self, node):
        self.scope_and_imports.append(node)
        self.generic_visit(node)
        last_node = self.scope_and_imports.pop(-1)
        while isinstance(last_node, ImportInfo): 
            last_node = self.scope_and_imports.pop(-1)

            
root = ast.parse(open('/mnt/sunyanfang01/my_pytorch/Pytorch-UNet-torch/train.py').read())
convertor = Convertor(root)
convertor.run()

print(astor.to_source(convertor.root))
f = open("tmp.py", "w")
f.write(astor.to_source(convertor.root))
f.close()