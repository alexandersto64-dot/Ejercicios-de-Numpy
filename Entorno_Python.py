#Paso 1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#Paso 3
ventas = np.array([120,180,95,210,150])
print(ventas)
print(np.mean(ventas))
print(np.max(ventas))
print(np.min(ventas))
print(np.std(ventas))

#Aplicacion de vector
cantidad = np.array([10,5,8,12])
precio=np.array([50,80,65,40])
ingresos= cantidad * precio
print(ingresos)

#Paso 4
#Cantidad | Precio | Venta
datos = np.array([
    
    [10,50,500],
    [5,80,400],
    [12,40,480]
])
print(datos)

#Vectores, matrices y Machine Learning
X=[
    [10, 50, 30, 0.10],
    [15, 45, 28, 0.05],
    [8, 60, 35, 0.00]

]
Y= [20,27,15]

#PASO 5 — Comprender Pandas
#Representación unidimensiona
ventas1 = pd.Series([500, 700, 450, 900])
print(ventas1)

#DataFrame
datos1 = {
"Producto": ["A", "B", "C"],
"Cantidad": [10, 15, 8],
"Precio": [50, 45, 60]
}
df = pd.DataFrame(datos1)
print(df)

#PASO 6 — ETL: EXTRACT
#Leer Excel con Pandas
ventas = pd.read_excel(
"Ventas_2026.xlsx",
sheet_name="Ventas"
)
print(ventas.head())

#Analizar el archivo cargado

#Número de registros:
print(ventas.shape)
#Columnas:
print(ventas.columns)
#Información general:
ventas.info()
#Primeras observaciones:
ventas.head()
#Últimas observaciones:
ventas.tail()
#Resumen estadístico:
ventas.describe()

#PASO 7 — Leer CSV utilizando Pandas

#Cargamos inventario:
inventario = pd.read_csv(
"inventario.csv"
)
inventario.head()

#PASO 8 — Leer CSV utilizando NumPy
inventario_numpy = np.genfromtxt(
"inventario.csv",
delimiter=",",
dtype=None,
names=True,
encoding="utf-8"
)
print(inventario_numpy)
#PASO 9 — ETL: TRANSFORM

#Revisar valores nulos
print(ventas.isnull().sum())
#Corregir valores faltantes
ventas["Cantidad"] = ventas["Cantidad"].fillna(
ventas["Cantidad"].median()
)
#Precio:
ventas["PrecioUnitario"] = ventas["PrecioUnitario"].fillna(
ventas["PrecioUnitario"].median()
)
#Costo:
ventas["CostoUnitario"] = ventas["CostoUnitario"].fillna(
ventas["CostoUnitario"].median()
)

#Eliminar registros duplicados
#Primero verificamos:
ventas.duplicated().sum()
#Después eliminamos:
ventas = ventas.drop_duplicates()

#Convertir la fecha correctamente
ventas["Fecha"] = pd.to_datetime(
ventas["Fecha"],
errors="coerce"
)

#Convertir variables numéricas
columnas_numericas = [
"Cantidad",
"PrecioUnitario",
"CostoUnitario",
"Descuento"
]
for columna in columnas_numericas:
    ventas[columna] = pd.to_numeric(
ventas[columna],
errors="coerce"
)
    
#PASO 10 — Crear nuevas variables mediante operaciones vectorizadas
#Calcularemos el importe bruto.
ventas["VentaBruta"] = (
ventas["Cantidad"] * ventas["PrecioUnitario"]
)
#Calcular descuento:
ventas["MontoDescuento"] = (
ventas["VentaBruta"] * ventas["Descuento"]
)
#Venta neta:
ventas["VentaNeta"] = (
ventas["VentaBruta"] - ventas["MontoDescuento"]
)
#Costo:
ventas["CostoTotal"] = (
ventas["Cantidad"] * ventas["CostoUnitario"]
)
#Utilidad:
ventas["Utilidad"] = (
ventas["VentaNeta"] - ventas["CostoTotal"]
)

#Aplicación del concepto de vector
ventas["Cantidad"] * ventas["PrecioUnitario"]

#PASO 11 — Crear variables temporales
#Extraemos información de las fechas.
ventas["Anio"] = ventas["Fecha"].dt.year
ventas["Mes"] = ventas["Fecha"].dt.month
ventas["Dia"] = ventas["Fecha"].dt.day
ventas["DiaSemana"] = ventas["Fecha"].dt.day_name()
#También podemos obtener la semana:
ventas["Semana"] = (
ventas["Fecha"]
.dt.isocalendar()
.week
)

#PASO 12 — Integrar ventas e inventario
datos = ventas.merge(
inventario,
on="ProductoID",
how="left"
)
#Comprobamos:
datos.head()

#PASO 13 — Análisis exploratorio de datos
#Ventas totales:
ventas_totales = datos["VentaNeta"].sum()
print(ventas_totales)
#Utilidad total:
utilidad_total = datos["Utilidad"].sum()
print(utilidad_total)
#Cantidad vendida:
cantidad_total = datos["Cantidad"].sum()
print(cantidad_total)

# Productos con mayor venta
ventas_producto = (
datos
.groupby("Producto")["VentaNeta"]
.sum()
.sort_values(ascending=False)
)
print(ventas_producto.head(10))

#Categorías con mayor rentabilidad
rentabilidad_categoria = (
datos
.groupby("Categoria")["Utilidad"]
.sum()
.sort_values(ascending=False)
)
print(rentabilidad_categoria)

#Analizar ventas por canal
ventas_canal = (
datos
.groupby("Canal")["VentaNeta"]
.sum()
.sort_values(ascending=False)
)
print(ventas_canal)

#PASO 14 — Detectar valores atípicos utilizando estadística
#Utilizaremos el rango intercuartílico.
Q1 = datos["Cantidad"].quantile(0.25)
Q3 = datos["Cantidad"].quantile(0.75)
IQR = Q3 - Q1
#Límites:
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR
#Detectar outliers:
outliers = datos[
(datos["Cantidad"] < limite_inferior) |
(datos["Cantidad"] > limite_superior)
]
print(outliers)

#¿Por qué identificar outliers?
#Un valor extremo podría representar:
#Error de digitación Venta corporativa Compra excepcional Promoción
#Campaña Estacionalidad Fraude Evento comercial
#En Ciencia de Datos no debemos eliminar automáticamente un outlier.
#Primero debemos investigarlo.

#PASO 15 — Preparar información para Machine Learning
#Análisis descriptivo
#hacia:
#Análisis predictivo.
#Nuestro objetivo será:
#Predecir la demanda de cada producto para la siguiente semana

#PASO 16 — Construir dataset semanal
#Agrupamos la información.
semanal = (
datos
.groupby(
["ProductoID", "Anio", "Semana"]
)
.agg(
CantidadVendida=("Cantidad", "sum"),
VentaNeta=("VentaNeta", "sum"),
PrecioPromedio=("PrecioUnitario", "mean"),
Utilidad=("Utilidad", "sum"),
StockActual=("StockActual", "mean"),
LeadTimeDias=("LeadTimeDias", "mean")
)
.reset_index()
)
#Ordenamos:
semanal = semanal.sort_values(
["ProductoID", "Anio", "Semana"]
)

#PASO 17 — Ingeniería de características
#Crearemos ventas de semanas anteriores.

#Lag 1
semanal["Lag1"] = (
semanal
.groupby("ProductoID")["CantidadVendida"]
.shift(1)
)

#Lag 2
semanal["Lag2"] = (
semanal
.groupby("ProductoID")["CantidadVendida"]
.shift(2)
)

#Lag 3
semanal["Lag3"] = (
semanal
.groupby("ProductoID")["CantidadVendida"]
.shift(3)
)
print("\n===== DATOS SEMANALES =====")
print(semanal)

print("\n===== VALORES NULOS =====")
print(semanal.isnull().sum())

print("\n===== FILAS ANTES DE DROPNA =====")
print(len(semanal))

modelo_df = semanal.dropna().copy()

print("\n===== FILAS DESPUÉS DE DROPNA =====")
print(len(modelo_df))
#Crear promedio móvil
#Calcularemos el promedio de demanda de las últimas cuatro semanas.
semanal["PromedioMovil4"] = (
semanal
.groupby("ProductoID")["CantidadVendida"]
.transform(
lambda x: x.shift(1).rolling(4).mean()
)
)

#Crear variable objetivo
#Nuestro algoritmo debe intentar predecir:
#Cantidad vendida la próxima semana
semanal["DemandaProximaSemana"] = (
semanal
.groupby("ProductoID")["CantidadVendida"]
.shift(-1)
)

#Eliminar registros incompletos
modelo_df = semanal.dropna().copy()
#Podemos comprobar:
modelo_df.head()

#PASO 18 — Definir variables X e y
#Variables independientes:
features = [
"CantidadVendida",
"Lag1",
"Lag2",
"Lag3",
"PromedioMovil4",
"PrecioPromedio",
"VentaNeta",
"Utilidad",
"StockActual",
"LeadTimeDias"
]
#Matriz X:
X = modelo_df[features]
#Vector y:
y = modelo_df["DemandaProximaSemana"]
print("\n===== COMPROBACIÓN DEL MODELO =====")
print("Registros semanales:", len(semanal))
print("Registros para ML:", len(modelo_df))
print("Valores nulos:")
print(modelo_df[features].isnull().sum())

#Relación con vectores y matrices
#Machine Learning utiliza matemáticamente
#Esta es una aplicación directa de vectores y matrices en Machine Learning

#PASO 19 — Separar entrenamiento y prueba
#Utilizaremos aproximadamente:
punto_corte=int(len(modelo_df)*0.80)
#Creamos:
X_train = X.iloc[:punto_corte]
X_test = X.iloc[punto_corte:]
y_train = y.iloc[:punto_corte]
y_test = y.iloc[punto_corte:]

#PASO 20 — Crear modelo de Machine Learning
#Random Forest Regressor : Es un algoritmo basado en múltiples árboles de decisión.
#Creamos el modelo:
modelo = RandomForestRegressor(
n_estimators=300,
max_depth=12,
random_state=42,
n_jobs=-1
)
#Entrenamos:
modelo.fit(
X_train,
y_train
)

# PASO 21 — Generar predicciones
predicciones = modelo.predict(X_test)
#Ahora tenemos:
#Demanda real
#versus:
#Demanda estimada

# PASO 22 — Evaluar el modelo MAE
# Mean Absolute Error:
mae = mean_absolute_error(
y_test,
predicciones
)
print("MAE:", mae)
# El MAE muestra aproximadamente cuántas unidades se equivoca el modelo.

#RMSE
rmse = np.sqrt(
mean_squared_error(
y_test,
predicciones
)
)
print("RMSE:", rmse)

#Coeficiente R²
r2 = r2_score(
y_test,
predicciones
)
print("R²:", r2)

#PASO 23 — Crear DataFrame de resultados
resultado = modelo_df.iloc[
punto_corte:
].copy()
#Agregamos:
resultado["DemandaPredicha"] = predicciones
#Creamos error:
resultado["ErrorAbsoluto"] = abs(
resultado["DemandaProximaSemana"] -
resultado["DemandaPredicha"]
)

#PASO 24 — Identificar riesgo de inventario
resultado["RiesgoStock"] = np.where(
resultado["DemandaPredicha"] > resultado["StockActual"],
"ALTO",
"NORMAL"
)

#Crear recomendación de reposición
resultado["CantidadSugeridaCompra"] = np.maximum(
 resultado["DemandaPredicha"] - resultado["StockActual"],
 0
)
#Podemos agregar un pequeño stock de seguridad:
resultado["StockSeguridad"] = (
 resultado["DemandaPredicha"] * 0.20
)
#Cantidad sugerida:
resultado["CompraRecomendada"] = np.maximum(
(
resultado["DemandaPredicha"] +
resultado["StockSeguridad"]
) - resultado["StockActual"],
0
)

# PASO 25 — Analizar importancia de variables
#Random Forest permite determinar qué variables tuvieron mayor influencia.
importancia = pd.DataFrame({
"Variable": features,
"Importancia": modelo.feature_importances_
})
#Ordenamos:
importancia = importancia.sort_values(
"Importancia",
ascending=False
)
print(importancia)

#PASO 26 — ETL: LOAD
# Exportaremos la información transformada. Dataset general:
datos.to_csv(
"dataset_powerbi.csv",
index=False,
encoding="utf-8-sig"
)
#Predicciones:
resultado.to_csv(
"predicciones_demanda.csv",
index=False,
encoding="utf-8-sig"
)
#Importancia de variables:
importancia.to_csv(
"importancia_variables.csv",
index=False,
encoding="utf-8-sig"
)

# Resultado del proceso ETL
#Partimos de:
#Ventas_2026.xlsx
#más:
#inventario.csv
#Python generó:
#dataset_powerbi.csv
#predicciones_demanda.csv
#importancia_variables.csv

#PASO 27 — Abrir Power BI
#Abrir:
#Power BI Desktop
#Seleccionar:
#Inicio → Obtener datos → Texto/CSV
#Importar:
#dataset_powerbi.csv
#Posteriormente importar:
#predicciones_demanda.csv
#Finalmente:
#importancia_variables.csv

# PASO 28 — Verificar los tipos de datos
#Desde Power Query comprobar:
#Fecha → Fecha.
#Cantidad → Número entero.
#Precio → Decimal.
#VentaNeta → Decimal.
#Utilidad → Decimal.
#StockActual → Número.
#DemandaPredicha → Decimal.
#Semana → Número entero.
#Seleccionar:
#Cerrar y aplicar.

#PASO 29 — Crear indicadores DAX

#

#

#

#