import pandas as pd

# We load Google stock data in a DataFrame
Google_stock = pd.read_csv('./goog-1.csv')

# We print some information about Google_stock
print('Google_stock is of type:', type(Google_stock))
print('Google_stock has shape:', Google_stock.shape)

#5 first
print(Google_stock.head())

#5 last
print(Google_stock.tail())

#N first
print(Google_stock.head(10))

#N last
print(Google_stock.tail(10))

print(Google_stock.isnull().any())

# We get descriptive statistics on our stock data
print()
print("DESCRIBE()")
print(Google_stock.describe())

# We get descriptive statistics on a single column of our DataFrame
print()
print("DESCRIBE single column")
print(Google_stock['Adj Close'].describe())


# We print information about our DataFrame  
print()
print('Maximum values of each column:\n', Google_stock.max())
print()
print('Minimum Close value:', Google_stock['Close'].min())
print()
print('Average value of each column:\n', Google_stock.mean())


# We display the correlation between columns
print()
print("Correlation")
print(Google_stock.corr())