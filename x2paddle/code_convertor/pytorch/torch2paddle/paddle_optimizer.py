from paddle.optimizer import SGD as Base_SGD
from paddle.regularizer import L2Decay

class SGD(Base_SGD):
    def __init__(self, 
                 learning_rate=0.001,
                 parameters=None,
                 weight_decay=None,
                 grad_clip=None,
                 name=None):
        parameters_list = list()
        if parameters is not None:
            for items in parameters:
                if isinstance(items, dict):
                    params = items["params"]
                    if "lr" in items:
                        for p in params:
                            p.optimize_attr["learning_rate"] = items["lr"] / learning_rate
                    if "weight_decay" in items:
                        for p in params:
                            if isinstance(items["weight_decay"], float):
                                p.regularizer = L2Decay(items["weight_decay"])
                            else:
                                p.regularizer = weight_decay
                    parameters_list.extend(params)
                    
                else:
                    parameters_list.append(items)
        super(SGD, self).__init__(
            learning_rate=learning_rate,
            parameters=parameters_list,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            name=name)