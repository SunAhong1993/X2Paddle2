import paddle
from functools import partial

def add_tensor_function(func):
    setattr(paddle.Tensor, func.__name__, func)

@add_tensor_function
def item(self):
    return self.numpy()[0]

@add_tensor_function
def permute(self, *dims):
    return self.transpose(dims)

@add_tensor_function
def contiguous(self):
    return self

@add_tensor_function
def view(self, *shape):
    return self.reshape(*shape)

@add_tensor_function
def dim(self):
    return self.ndim

@add_tensor_function
def long(self, memory_format=None):
    return paddle.cast(self, dtype="int64")

@add_tensor_function
def size(self, dim=None):
    if dim is not None:
        return self.shape[dim]
    else:
        return paddle.shape(self)

@add_tensor_function
def to(self, *args, **kwargs):
    if len(args) == 1 and "dtype" not in kwargs:
        try:
            return paddle.cast(self, dtype=args[0])
        except Exception:
            return self
    else:
        if len(kwargs) > 0:
            if "dtype" in kwargs:
                return paddle.cast(self, dtype=kwargs["dtype"])
            else:
                return self
        else:
            return self
        
@add_tensor_function
def index_fill_(self, dim, index, val):
    x_shape = self.shape
    index_shape = index.shape
    if dim != 0:
        perm_list = list(range(len(x_shape)))
        while dim < 0:
            dim += len(x_shape)
        perm_list.pop(dim)
        perm_list = [dim] + perm_list
        self = paddle.transpose(self, perm=perm_list)
        s = x_shape.pop(dim)
        x_shape = [s] + x_shape
    updates_shape = index_shape + x_shape[1:]
    updates = paddle.full(updates_shape, fill_value=val, dtype=self.dtype)
    out = paddle.scatter(self, index, updates)
    if dim != 0:
        perm_list = list(range(len(x_shape)))
        perm_list.pop(0)
        perm_list.insert(dim, 0)
        out = paddle.transpose(out, perm=perm_list)
    paddle.assign(out, output=self)


sum_tmp = partial(paddle.Tensor.sum)
@add_tensor_function
def sum(self, dim, keepdim=False, dtype=None):
    return sum_tmp(self, axis=dim, dtype=dtype, keepdim=keepdim)

sort_tmp = partial(paddle.Tensor.sort)
@add_tensor_function
def sort(self, dim=-1, descending=False, out=None):
    return sort_tmp(self, axis=dim, descending=descending), paddle.argsort(self, axis=dim, descending=descending)

sort_reshape = partial(paddle.Tensor.reshape)
@add_tensor_function
def reshape(self, *shape):
    return sort_reshape(self, shape)
