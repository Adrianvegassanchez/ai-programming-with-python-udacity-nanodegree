import numpy as np

#1D
x = np.array([1, 2, 3, 4, 5])
print()
print('x = ', x)
print()

# print information about x
print('x has dimensions:', x.shape)
print('x is an object of type:', type(x))
print('The elements in x are of type:', x.dtype)


# create a rank 2 ndarray
y = np.array([[1,2,3],[4,5,6],[7,8,9], [10,11,12]])

# print Y
print()
print('Y = \n', y)
print()

# information about Y
print('Y has dimensions:', y.shape)
print('Y has a total of', y.size, 'elements')
print('Y is an object of type:', type(y))
print('The elements in Y are of type:', y.dtype)

# save x into the current directory as 
np.save('my_array', x)

# load the saved array from our current directory into variable y
y = np.load('my_array.npy')

# print y
print()
print('y_loaded = ', y)
print()

# print information about the ndarray we loaded
print('y_loaded is an object of type:', type(y))
print('The elements in y_loaded are of type:', y.dtype)

