# Notes of PyTorch documentation

PyTorch (pt) is a Python-based scientific computing package serving two broad purposes:

- a replacement for NumPy to use the power of GPUs and other accelerators.
- An automatic differentiation library that is useful to implement neural networks

## PyTorch basic DL components

- Tensors : N-dimensional array that serve as pt fundamental data structure. They support automatic differentiation, hardware acceleration, and provide a comprehenisve API for mathematical operations.

- Autograd : pt automatic differentiation engine that tracks operations performed on tensors and builds a computatioanl graph dynamically to be able to compute gradients.

- NN API : A modular framework for building NN with pre-defined layers, activations functions and loss functions. The **nn.Module** base class provides a clean interface for creating custom network architecturs with parameter management.

- DataLoaders : Tools for efficient data handling that provide features like batching, shuffling and parallel data loading. They abstract away the complexities of data preprocessing and iteration, allowing for optimized training loops.

## Tensors

Tensors can be created directly from data our np arrays.

Tensor attributes can be used to describe their shape, datatype but most importantly also the device on which they are stored.

All the tensor operations will run on the GPU if the device is selected.

All operatiosn with a _ suffix will change the actual data.

## Autograd



## PyTorch Compiler

**torch.compiler** is a namespace through which some of the internal compiler methods are surfaced for user consumption. The main function and the feature in this namespace is **torch.compiler**.

You can use the compiler to select different backends to increase training & inference time or inference-only time see [reference table](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler.html)

```
torch.compile(m, backend="ipex")
```

Uses IPEX for inference on CPU, could be interessting. **-> only on Intel CPU**
