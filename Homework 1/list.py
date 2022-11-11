list_1 = [i**2 for i in range(100)]

list_2 = [i**2 for i in range(100) if i**2 % 2 == 0]

list_3 = [i for i in list if i**2 % 2 == 0]

list_2 == list_3
