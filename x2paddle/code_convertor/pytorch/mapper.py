from x2paddle.code_convertor.pytorch.api_mapper import *
from x2paddle.utils import *


OPTIMIZER_MAPPER = {"torch.optim": ["paddle.optimizer", None],
                    "torch.optim.lr_scheduler.ReduceLROnPlateau": ["paddle.optimizer.lr.ReduceOnPlateau", ClassReduceOnPlateau],
                    "torch.optim.Adam": ["paddle.optimizer.Adam", ClassAdam],}

NN_MAPPER = {"torch.nn": ["paddle.nn", None],
             "torch.nn.Conv2d": ["paddle.nn.Conv2D", ClassConv2d],
             "torch.nn.CrossEntropyLoss": ["paddle.nn.CrossEntropyLoss", ClassCrossEntropyLoss],
             "torch.nn.BCEWithLogitsLoss": ["paddle.nn.BCEWithLogitsLoss", ClassBCEWithLogitsLoss],
             "torch.nn.utils": ["paddle.nn.utils", None],
             "torch.nn.utils.clip_grad_value_": ["x2paddle.torch2paddle.clip_grad_value_", None]}

UTILS_MAPPER = {"torch.utils.data": ["paddle.io", None],
                "torch.utils.data.DataLoader": ["x2paddle.torch2paddle.BaseDataLoader", ClassDataLoader], 
                "torch.utils.data.random_split": ["x2paddle.torch2paddle.random_split", None]}

DTYPE_MAPPER = {"torch.float32": [string("float32"), None],
                "torch.long": [string("int64"), None]}

API_MAPPER = {"torch": ["paddle", None],
              "torch.load": ["paddle.load", FuncLoad],
              "torch.save": ["paddle.save", FuncSave],
              "torch.device": ["paddle.set_device", FuncSetDevice],
              "torch.cuda.is_available": ["paddle.is_compiled_with_cuda", None]}

API_MAPPER.update(OPTIMIZER_MAPPER)
API_MAPPER.update(NN_MAPPER)
API_MAPPER.update(UTILS_MAPPER) 
API_MAPPER.update(DTYPE_MAPPER) 
