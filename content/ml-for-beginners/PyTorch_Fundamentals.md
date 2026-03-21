# Notebook: PyTorch_Fundamentals

> Source: https://github.com/microsoft/ML-For-Beginners/blob/HEAD/PyTorch_Fundamentals.ipynb

---

<a href="https://colab.research.google.com/github/karthiksivakoti/ML-For-Beginners/blob/main/PyTorch_Fundamentals.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

```python
import torch
torch.__version__
```

```python
print("I am excited to run this")
```

```python
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
print(torch.__version__)
```

# **Introduction to Tensors**

```python
# scalar
scalar = torch.tensor(7)
scalar
```

```python
scalar.ndim
```

```python
scalar.item()
```

```python
# vector
vector = torch.tensor([7, 7])
vector
#vector.ndim
#vector.item()
```

```python
vector.shape
```

```python
# Matrix
MATRIX = torch.tensor([[7, 8],[9, 10]])
MATRIX
```

```python
MATRIX.ndim
```

```python
MATRIX[0]
MATRIX[1]
```

```python
# Tensor
TENSOR = torch.tensor([[[1, 2, 3],[3,6,9], [2,4,5]]])
TENSOR
```

```python
TENSOR.shape
```

```python
TENSOR.ndim
```

```python
TENSOR[0]
```

# RANDOM TENSOR

```python
random_tensor = torch.rand(3,4)
random_tensor
```

```python
random_tensor.ndim
```

```python
random_tensor.shape
```

```python
random_tensor.size()
```

```python
random_image_tensor = torch.rand(size=(3, 224, 224)) #color channels, height, width
random_image_tensor.ndim, random_image_tensor.shape
```

```python
random_tensor_ofownsize = torch.rand(size=(5,10,10))
random_tensor_ofownsize.ndim, random_tensor_ofownsize.shape

```

Zeroes and Ones tensor

```python
zero = torch.zeros(size=(3, 4))
zero
```

```python
zero*random_tensor
```

```python
ones = torch.ones(size=(3, 4))
ones

```

```python
ones.dtype
```

```python
ones*zero
```

Range of Tensors, Tensor - like

```python
one_to_ten = torch.arange(start = 1, end = 11, step = 1)
one_to_ten
```

```python
ten_zeros = torch.zeros_like(one_to_ten)
ten_zeros
```

Tensor Datatypes

```python
float_32_tensor = torch.tensor([3.0, 6.0,9.0], dtype = None, device = None, requires_grad = False)
float_32_tensor
```

```python
float_32_tensor.dtype
```

```python
float_16_tensor = float_32_tensor.type(torch.float16)
float_16_tensor.dtype
```

```python
float_16_tensor*float_32_tensor
```

```python
int_32_tensor = torch.tensor([3, 6, 9], dtype = torch.int32)
int_32_tensor
```

```python
int_32_tensor*float_32_tensor
```

```python
x = torch.arange(0,100,10)
```

```python
x
```

```python
x.min()
```

```python
x.max()
```

```python
torch.mean(x.type(torch.float32))
```

```python
x.type(torch.float32).mean()
```

```python
x.sum()
```

```python
x.argmax()
```

```python
x.argmin()
```

```python
x[0]
```

```python
x[9]
```

```python
x = torch.arange(1, 10)
x.shape
```

```python
x_reshaped = x.reshape(1,9)
x_reshaped, x_reshaped.shape
```

```python
x_reshaped.view(1,9)
```

```python
x_stacked = torch.stack([x, x, x, x], dim = 1)
x_stacked
```

```python
x_stacked.squeeze()
```

```python
x_stacked.unsqueeze(dim=1)
```

```python
x_stacked.squeeze()
```

```python
x_stacked.unsqueeze(dim=-2)
```

```python
import torch
tensor = torch.tensor([1, 2, 3])
tensor = tensor - 10
tensor
```

```python
torch.mul(tensor, 10)
```

```python
torch.sub(tensor, 100)
```

```python
torch.add(tensor, 100)
```

```python
torch.divide(tensor, 2)
```

```python
torch.matmul(tensor, tensor)
```

```python
tensor@tensor
```

```python
%%time
tensor@tensor
```

```python
%%time
torch.matmul(tensor,tensor)
```

```python
torch.rand(3,2)
```

```python
torch.matmul(torch.rand(3,2), torch.rand(2,3))
```

```python
import torch
```

```python
x = torch.rand(2,9)
```

```python
x
```

```python
y=torch.randn(2,3,5)
y
```

```python
x_original = torch.rand(size=(224,224,3))
x_original
```

```python
x_permuted=x_original.permute(2, 0, 1)
print(x_original.shape)
print(x_permuted.shape)
```

```python
x_original[0,0,0]
```

```python
x_permuted[0,0,0]
```

```python
x_original[0,0,0]=0.989
```

```python
x_original[0,0,0]
```

```python
x_permuted[0,0,0]
```

```python
x=torch.arange(1,10).reshape(1,3,3)
x, x.shape
```

```python
x[0]
```

```python
x[0][0]
```

```python
x[0][0][0]
```

```python
x[0][2][2]
```

```python
x[:,1,1]
```

```python
x[0,0,:]
```

```python
x[0,:,2]
```

```python
import numpy as np
```

```python
array = np.arange(1.0, 8.0)
```

```python
array
```

```python
tensor = torch.from_numpy(array)
tensor
```

```python
array[3]=11.0
```

```python
array
```

```python
tensor
```

```python
tensor = torch.ones(7)
tensor, tensor.dtype
numpy_tensor = tensor.numpy()
numpy_tensor, numpy_tensor.dtype
```

```python
import torch
random_tensor_A = torch.rand(3,4)
random_tensor_B = torch.rand(3,4)
print(random_tensor_A)
print(random_tensor_B)
print(random_tensor_A == random_tensor_B)
```

```python
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
random_tensor_C = torch.rand(3,4)
torch.manual_seed(RANDOM_SEED)
random_tensor_D = torch.rand(3,4)
print(random_tensor_C)
print(random_tensor_D)
print(random_tensor_C == random_tensor_D)
```

```python
!nvidia-smi
```

```python
import torch
torch.cuda.is_available()
```

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
device
```

```python
torch.cuda.device_count()
```

```python
tensor = torch.tensor([1,2,3], device = "cpu")
print(tensor, tensor.device)
```

```python
tensor_on_gpu = tensor.to(device)
tensor_on_gpu
```

```python
tensor_on_gpu.numpy()
```

```python
tensor_on_cpu = tensor_on_gpu.cpu().numpy()
```