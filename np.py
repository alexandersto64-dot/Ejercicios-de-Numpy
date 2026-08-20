import numpy as np

numeros = np.array([10, 20, 30, 40, 50])
print(numeros)

#Acceder a los elementos del array
print(numeros[0])
print(numeros[1])
print(numeros[2])
print(numeros[3])
print(numeros[4])

#Operaciones basicos
print(numeros + 10)
print(numeros**2)

#Estadisticas
print("Suma: " , np.sum(numeros))
print("Promedio: " , np.mean(numeros))
print("Maximo: " , np.max(numeros))
print("Minimo: " , np.min(numeros))

#Matrices
matriz = np.array([
    [10,20,30],
    [40,50,60]
])
print(matriz)

print(matriz[0])
print(matriz[:,1])
print(matriz.shape)

#ventas
ventas = np.array([500,1200,800,2000,1500])
print(np.sort(ventas))
print(ventas[ventas>1000])
resultado = np.where(ventas >1000, "Meta alcanzada","No alcanzo meta")
print(resultado)

A = np.array([
    [1,2],
    [3,4]
])

B = np.array([
    [5,6],
    [7,8]
])

print(A+B)

#Axis para suma y otros funciones para columnas vertical y horizontal
ventas1 = np.array([
    [100,200,300],
    [150,250,350],
    [200,300,400]
])

print("Por columna: " ,  np.mean(ventas1, axis=1))