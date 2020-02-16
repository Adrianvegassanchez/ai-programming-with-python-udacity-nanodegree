import pandas as pd

# We create a Pandas Series that stores a grocery list of just fruits
fruits = pd.Series(data = [10, 6, 3,], index = ['apples', 'oranges', 'bananas'])

# We display the fruits Pandas Series
print(fruits)

#basic operations :

# We print fruits for reference
print('Original grocery list of fruits:\n ', fruits)

# We perform basic element-wise operations using arithmetic symbols
print()
print('fruits + 2:\n', fruits + 2) # We add 2 to each item in fruits
print()
print('fruits - 2:\n', fruits - 2) # We subtract 2 to each item in fruits
print()
print('fruits * 2:\n', fruits * 2) # We multiply each item in fruits by 2 
print()
print('fruits / 2:\n', fruits / 2) # We divide each item in fruits by 2
print()

#Mathematical operations

# We print fruits for reference
print('Original grocery list of fruits:\n', fruits)

# We apply different mathematical functions to all elements of fruits
print()
print('EXP(X) = \n', pd.exp(fruits))
print() 
print('SQRT(X) =\n', pd.sqrt(fruits))
print()
print('POW(X,2) =\n', pd.power(fruits,2)) # We raise all elements of fruits to the power of 2