from x2paddle.code_convertor.pytorch.api_mapper import *
from x2paddle.utils import *


OPTIMIZER_MAPPER = {"torch.optim": ["paddle.optimizer", None],
                    "torch.optim.lr_scheduler.ReduceLROnPlateau": ["paddle.optimizer.lr.ReduceOnPlateau", ClassReduceOnPlateau],
                    "torch.optim.lr_scheduler.CosineAnnealingLR": ["paddle.optimizer.lr.CosineAnnealingDecay", 
                                                                   ClassCosineAnnealingDecay],
                    "torch.optim.lr_scheduler.MultiStepLR": ["paddle.optimizer.lr.MultiStepDecay", ClassMultiStepDecay],
                    "torch.optim.Adam": ["x2paddle.torch2paddle.Adam", ClassAdam]}

NN_MAPPER = {"torch.nn": ["paddle.nn", None],
             "torch.nn.DataParallel": ["paddle.DataParallel", ClassDataParallel],
             "torch.nn.Module": ["paddle.nn.Layer", None],
             "torch.nn.Conv2d": ["paddle.nn.Conv2D", ClassConv2D],
             "torch.nn.ConvTranspose2d": ["paddle.nn.Conv2DTranspose", ClassConv2DConv2DTranspose],
             "torch.nn.BatchNorm2d": ["paddle.nn.BatchNorm2D", ClassBatchNorm2D],
             "torch.nn.MaxPool2d": ["paddle.nn.MaxPool2D", ClassMaxPool2D],
             "torch.nn.Upsample": ["paddle.nn.Upsample", None],
             "torch.nn.ReLU": ["paddle.nn.ReLU", ClassReLU],
             "torch.nn.CrossEntropyLoss": ["paddle.nn.CrossEntropyLoss", ClassCrossEntropyLoss],
             "torch.nn.BCEWithLogitsLoss": ["paddle.nn.BCEWithLogitsLoss", ClassBCEWithLogitsLoss],
             "torch.nn.Sequential":["paddle.nn.Sequential", None],
             "torch.nn.utils": ["paddle.nn.utils", None],
             "torch.nn.utils.clip_grad_value_": ["x2paddle.torch2paddle.clip_grad_value_", None],
             "torch.nn.functional": ["paddle.nn.functional", None],
             "torch.nn.functional.pad": ["paddle.nn.functional.pad", FuncPad],
             "torch.nn.functional.cross_entropy": ["paddle.nn.functional.cross_entropy", FuncCrossEntropy],
             "torch.nn.functional.softmax": ["paddle.nn.functional.softmax", FuncSoftmax],
             "torch.sigmoid": ["paddle.nn.functional.sigmoid", FuncSigmoid],}

UTILS_MAPPER = {"torch.utils.data": ["paddle.io", None],
                "torch.utils.data.DataLoader": ["x2paddle.torch2paddle.BaseDataLoader", ClassDataLoader], 
                "torch.utils.data.random_split": ["x2paddle.torch2paddle.random_split", None],
                "torch.utils.data.Dataset": ["paddle.io.Dataset", None],
                "torch.utils.data.ConcatDataset": ["x2paddle.torch2paddle.ConcatDataset", None]}

DTYPE_MAPPER = {"torch.float32": [string("float32"), None],
                "torch.long": [string("int64"), None]}

TORCHVISION_MAPPER  = {"torchvision.transforms": ["paddle.vision.transforms", None],
                       "torchvision.transforms.Compose": ["paddle.vision.transforms.Compose", None],
                       "torchvision.transforms.ToPILImage": ["x2paddle.torch2paddle.ToPILImage", None],
                       "torchvision.transforms.Resize": ["paddle.vision.transforms.Resize", None],
                       "torchvision.transforms.ToTensor": ["paddle.vision.transforms.ToTensor", None]}

API_MAPPER = {"torch": ["paddle", None],
              "torch.load": ["paddle.load", FuncLoad],
              "torch.save": ["paddle.save", FuncSave],
              "torch.device": ["paddle.set_device", FuncSetDevice],
              "torch.cat": ["paddle.concat", FuncConcat],
              "torch.cuda.is_available": ["paddle.is_compiled_with_cuda", None],
              "torch.no_grad": ["paddle.no_grad", None],
              "torch.from_numpy": ["paddle.to_tensor", None],
              "torch.cuda.device_count": ["x2paddle.torch2paddle.device_count", None]}

API_MAPPER.update(OPTIMIZER_MAPPER)
API_MAPPER.update(NN_MAPPER)
API_MAPPER.update(UTILS_MAPPER) 
API_MAPPER.update(DTYPE_MAPPER) 
API_MAPPER.update(TORCHVISION_MAPPER)
