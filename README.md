# RepasoFinalDL


## SGD class (1 point)

This class is contained in the src.optimization module. This class is a custom implementation of SGD algorithm. The recommendation is to look at the documentation: link. To implement it you are not allowed to use pytorch methods (e.g. add, mul, div, etc).

## NAdam algorithm (1 point)

This class is contained in the src.optimization module. You will have to complete the constructor and step method. It is not allowed to use methods such as add or sub, only +, -, * and /. 
It is recommended to look at the steps of the algorithm in the PyTorch documentation.

## GroupNorm (1 point)

Development of the forward pass of the GroupNorm layer. No for-loops or functions from the nn package
can be used.
You can use the torch functions, but you should avoid using the functions that are not allowed (e.g. add, sub, mul, etc).

## Adam class (1 point)

This class is contained in the src.optimization module. This class is a custom implementation of Adam algorithm. The recommendation is to look at the documentation: link. To implement it you are not allowed to use pytorch methods (e.g. add, mul, div, etc).

## GroupNorm (1 point)

Development of the forward pass of the GroupNorm layer. No for-loops or functions from the nn package can be used.
You can use the torch functions, but you should avoid using the functions that are not allowed (e.g. add, sub, mul, etc).
## MaxPool2d (3 points)

Respect to the MaxPool2d function:
1. Development of an adaptation of fold and unfold to use them with the MaxPool2d. No for-loops
or functions from the nn package can be used, except fold and unfold nn function to code the
adaptations. (1 point)
2. Development of forward and backward of MaxPool2d. To code it, you have to use the adaptations
fold and unfold coded before, no other implementation will be allowed. No for-loops or functions
from the nn package can be used. (2 points )