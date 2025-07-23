import numpy as np

tensor_constructor = np.array
tensor_class = np.ndarray

# op functions
absolute = np.absolute
all = np.all
any = np.any
argmax = np.argmax
argmin = np.argmin
argwhere = np.argwhere
atleast_1d = np.atleast_1d
atleast_2d = np.atleast_2d
atleast_3d = np.atleast_3d
ceil = np.ceil
copy = np.copy
cos = np.cos
cosh = np.cosh
exp = np.exp


def flatten(a, order="C"):
    return a.flatten(order=order)


flip = np.flip
floor = np.floor
invert = np.invert
log = np.log
logical_not = np.logical_not
max = np.max
mean = np.mean
min = np.min
prod = np.prod


def ravel(a, order="C"):
    return a.ravel(order=order)


sign = np.sign
sin = np.sin
sinh = np.sinh
squeeze = np.squeeze
sum = np.sum
tan = np.tan
tanh = np.tanh
transpose = np.transpose
add = np.add
astype = np.astype
broadcast_to = np.broadcast_to
dot = np.dot
equal = np.equal
expand_dims = np.expand_dims
floor_divide = np.floor_divide


def getitem(a, key):
    return a[key]


greater = np.greater
greater_equal = np.greater_equal
less = np.less
less_equal = np.less_equal
logical_and = np.logical_and
logical_or = np.logical_or
logical_xor = np.logical_xor
matmul = np.matmul
mod = np.mod
multiply = np.multiply
not_equal = np.not_equal
power = np.power
reshape = np.reshape
subtract = np.subtract
tensordot = np.tensordot
true_divide = np.true_divide
clip = np.clip
swapaxes = np.swapaxes
where = np.where

# tensor functions
ones_like = np.ones_like
ones = np.ones
zeros_like = np.zeros_like
zeros = np.zeros
full_like = np.full_like
full = np.full
index_add = np.add.at
isin = np.isin
unravel_index = np.unravel_index
take_along_axis = np.take_along_axis
put_along_axis = np.put_along_axis
repeat = np.repeat
tile = np.tile
arange = np.arange
stack = np.stack
save = np.save
load = np.load
choice = np.random.choice
rand = np.random.rand
randint = np.random.randint
randn = np.random.randn
binomial = np.random.binomial
permutation = np.random.permutation
shuffle = np.random.shuffle
split = np.split

# dtypes
dtype = np.dtype
float64 = np.float64
float32 = np.float32
float16 = np.float16
uint64 = np.uint64
uint32 = np.uint32
uint16 = np.uint16
uint8 = np.uint8
int64 = np.int64
int32 = np.int32
int16 = np.int16
int8 = np.int8
bool = np.bool
