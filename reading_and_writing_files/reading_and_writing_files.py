f_write = open('file.txt', 'w') #w to write
f_write.write('hi form py!')
f_write.close()

f_read = open('file.txt', 'r')
file_data = f_read.read()
f_read.close()


print(file_data)