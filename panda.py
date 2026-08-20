import pandas as pd

notas = pd.Series([15, 18, 12, 16, 20])

print(notas)

datos = {
    "Nombre": ["Ana", "Juan", "Teresa", "Solar"],
    "Edad": [20, 21, 22, 23],
    "Notas": [15, 18, 14, 17]
}

df = pd.DataFrame(datos)

print(df)

print(df["Nombre"])

print(df[["Nombre", "Edad"]])

print(df.iloc[0])

# Leer archivo CSV
df = pd.read_csv("Empleados.csv", sep=";")

print(df)

# Rellenar valores vacíos de Nombre
df["Nombre"] = df["Nombre"].fillna(0)

print(df.isnull())