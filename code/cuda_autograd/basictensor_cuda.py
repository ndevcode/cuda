import numpy as np
import my_cuda_ops

class BasicTensor :
    """Stores a single BasicTensor value and its gradient"""

    def __init__(self, value, _children=(), _what=""):
        self.value = np.array(np.clip(value,-1e10,1e10),dtype=np.float32)
        self.grad = np.zeros_like(self.value)
        self._backward = lambda : None
        self._prev = set(_children)
        self._what = _what

    def __len__(self) :
        return len(self.value)

    @staticmethod
    def _broadcast_axes(out_shape,orig_shape) : 
        ndim_out = len(out_shape)
        ndim_orig = len(orig_shape)
        axes = []
        for i in range(ndim_out):
            j = i - (ndim_out - ndim_orig)  # corresponding axis in orig_shape
            if j < 0 or orig_shape[j] == 1:
                axes.append(i)
        return tuple(axes)

    def __add__(self,other) :
        other = other if isinstance(other,BasicTensor) else BasicTensor(other)
        out = BasicTensor(value = self.value + other.value, _children = (self,other), _what='+')

        def _backward() :
            axes_self = BasicTensor._broadcast_axes(out.value.shape, self.value.shape)
            self.grad += np.sum(out.grad, axis = axes_self, keepdims = True).reshape(self.value.shape)

            axes_other = BasicTensor._broadcast_axes(out.value.shape, other.value.shape)
            other.grad += np.sum(out.grad, axis=axes_other , keepdims=True).reshape(other.value.shape)
        out._backward = _backward

        return out

    def __mul__(self,other) :
        other = other if isinstance(other,BasicTensor) else BasicTensor(other)
        out = BasicTensor(value = self.value * other.value, _children = (self,other), _what='*')

        def _backward() :
            tmp_other = out.grad * other.value
            axes_self = BasicTensor._broadcast_axes(out.value.shape, self.value.shape)
            self.grad += np.sum(tmp_other, axis = axes_self, keepdims = True).reshape(self.value.shape)

            tmp_self = out.grad * self.value
            axes_other = BasicTensor._broadcast_axes(out.value.shape, other.value.shape)
            other.grad += np.sum(tmp_self, axis=axes_other , keepdims=True).reshape(other.value.shape)
        out._backward = _backward

        return out

    def __pow__(self,other) :
        assert (isinstance(other,(int,float))),"only support for int/float"
        out = BasicTensor(value = self.value ** other, _children = (self,), _what=f'**{other}')

        def _backward() :
            self.grad += (other * self.value ** (other-1)) * out.grad
        out._backward = _backward

        return out

    def __matmul__(self,other) :
        other = other if isinstance(other,BasicTensor) else BasicTensor(other)
        out = BasicTensor(value = self.value @ other.value, _children = (self,other), _what='@')

        def _backward() :
            self.grad += out.grad @ other.value.T
            other.grad += self.value.T @ out.grad if self.value.ndim > 1 else np.outer(self.value, out.grad)
        out._backward = _backward

        return out

    def __truediv__(self, other):
        other = other if isinstance(other, BasicTensor) else BasicTensor(other)
        out = BasicTensor(self.value / (other.value + 1e-12), _children=(self, other), _what="/")
    
        def _backward():
            eps = 1e-12  # to avoid division by zero
            # dL/dself = 1/other
            axes_self = BasicTensor._broadcast_axes(out.value.shape, self.value.shape)
            grad_self = out.grad / (other.value + eps)
            self.grad += np.sum(grad_self, axis=axes_self, keepdims=True).reshape(self.value.shape)
    
            # dL/dother = -(self / other^2)
            axes_other = BasicTensor._broadcast_axes(out.value.shape, other.value.shape)
            grad_other = -(self.value / ((other.value ** 2) + eps)) * out.grad
            other.grad += np.sum(grad_other, axis=axes_other, keepdims=True).reshape(other.value.shape)
    
        out._backward = _backward
        return out

    def relu(self):
        # Allocate output array of same shape
        out_value = np.empty_like(self.value)
        
        # Call your CUDA kernel (forward)
        out_value = my_cuda_ops.relu_forward(self.value)
        
        out = BasicTensor(out_value, _children=(self,), _what='ReLU')
    
        def _backward():
            grad_in = np.empty_like(self.grad)
            # Call your CUDA kernel (backward)
            grad_in = my_cuda_ops.relu_backward(self.value, out.grad)
            self.grad += grad_in
    
        out._backward = _backward
        return out

#     def relu(self) : 
#         out = BasicTensor(value = np.maximum(self.value, 0),_children=(self,), _what='ReLU')
# 
#         def _backward() :
#             self.grad += (self.value > 0) * out.grad
#         out._backward = _backward
# 
#         return out

    def backward(self) :
        #topological order
        topo = []
        visited = set()
        def doTopo(node) :
            if node not in visited :
                for child in node._prev :
                    doTopo(child)
                topo.append(node)
        doTopo(self)

        self.grad = np.ones_like(self.value)
        for node in reversed(topo) :
            node._backward()
    
    @property
    def T(self):
        out = BasicTensor(value = self.value.T, _children=(self,), _what='T')

        def _backward() :
            self.grad += out.grad.T
        out._backward = _backward

        return out

    def log(self) :
        out = BasicTensor(value = np.log(self.value + 1e-12), _children=(self,), _what='log')

        def _backward() :
            self.grad += (1.0/(self.value + 1e-12)) * out.grad
        out._backward = _backward

        return out

    def mean(self) :
        out = BasicTensor(np.mean(self.value), _children=(self,), _what='mean')
        def _backward() :
            self.grad += out.grad * np.ones_like(self.value) / self.value.size
        out._backward = _backward
        return out

    def exp(self) :
        out = BasicTensor(np.exp(self.value), _children=(self,), _what='exp')
        def _backward() :
            self.grad += out.grad * out.value
        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = BasicTensor(self.value.sum(axis=axis, keepdims=keepdims), _children=(self,), _what='sum')
        def _backward():
            # broadcast gradient back to input shape
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis)
            self.grad += np.ones_like(self.value) * grad
        out._backward = _backward
        return out

# In cross_entropy_loss
    def __repr__(self) :
        return f"BasicTensor(value={self.value}, grad = {self.grad})"


if __name__ == "__main__" :
    batch = 2
    nin = 3
    nout = 4

    X = BasicTensor(np.random.randn(batch, nin))
    W = BasicTensor(np.random.randn(nout, nin))  # note: weights shaped (nout, nin)
    b = BasicTensor(np.zeros(nout))

    Z = X @ W.T + b  # vectorized forward
    A = Z.relu()

    A.backward()
    print("X.grad:", X.grad)
    print("W.grad:", W.grad)
    print("b.grad:", b.grad)

    # simple scalar test
    x = BasicTensor(2.0)
    y = BasicTensor(3.0)
    z = x * y + x**2
    z.backward()
    print(x, y)
