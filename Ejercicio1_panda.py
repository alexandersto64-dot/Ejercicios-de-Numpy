import pandas as pd
import numpy as np

# Leer el archivo CSV
df = pd.read_csv("ventas_empresa.csv")
print(df)

# Mostrar los primeros registros
print(df.head())

# Mostrar cantidad de filas y columnas
print("Filas y columnas:", df.shape)

# Calcular una columna llamada Ingreso
df["Ingreso"] = df["Cantidad"] * df["Precio"]

print(df)

# Determinar el ingreso total
print("Ingreso total:", df["Ingreso"].sum())

# Calcular el precio promedio
print("Precio promedio:", df["Precio"].mean())

# Determinar el producto más caro
print("Producto más caro:", df.loc[df["Precio"].idxmax(), "Producto"])

# Determinar el producto más vendido
print("Producto más vendido:", df.loc[df["Cantidad"].idxmax(), "Producto"])

# Filtrar productos cuyo ingreso sea mayor a S/ 3000
print("Productos con ingreso mayor a S/ 3000:")
print(df[df["Ingreso"] > 3000])

# Convertir Cantidad en un array NumPy
cantidades = np.array(df["Cantidad"])

print("Array de cantidades:")
print(cantidades)

# Calcular máximo, mínimo, promedio y suma utilizando NumPy
print("Máximo:", np.max(cantidades))
print("Mínimo:", np.min(cantidades))
print("Promedio:", np.mean(cantidades))
print("Suma:", np.sum(cantidades))

# Array de NumPy

array = np.array([10, 20, 30, 40])
print(array)

# DataFrame de Pandas

df = pd.DataFrame({
    "Nombre": ["Ana", "Juan"],
    "Edad": [20, 21]
})
print(df)

matriz = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
print(matriz)