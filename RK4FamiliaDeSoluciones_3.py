import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import symbols, Function, dsolve, exp, sin, pi, tan, sympify

def funcion_f(ecuacion:str="x"):
    function = sp.sympify(ecuacion)
    return function

# Se define la familia de funciones
def FamiliaSoluciones(C:1):
    # Se define cual va a hacer la variable independiente (t)
    t = sp.symbols('t')
    solucionGeneral = 2 * sp.atan(C * sp.exp(sp.exp(t)))
    return solucionGeneral

# Estas constantes de integración (C) se obtinen al despejar la famlia de soluciones con el valor inicial
C_1 = ((np.tan(3/20)) / (sp.exp(sp.exp(3/10))))
C_2 = ((np.tan(301/2000)) / (sp.exp(sp.exp(301/1000))))
C_3 = ((np.tan(151/1000)) / (sp.exp(sp.exp(151/500))))

solucionSimbolica_1 = FamiliaSoluciones(C_1)
solucionSimbolica_2 = FamiliaSoluciones(C_2)
solucionSimbolica_3 = FamiliaSoluciones(C_3)

# Dijita el número de iteraciones:
numeroIteraciones = 8
# Dijita el valor inicial en el eje t:
pasoInicial_t = -4
# Dijita el valor final en el eje t:
pasoFinal_t = 4

delta_t = (pasoFinal_t - pasoInicial_t) / numeroIteraciones
print(delta_t)

# Creamos una lista con un solo elemento de nuestro valor inicial de t
vectorSolucion_t = np.array([pasoInicial_t])

# Iteramos para poblar el vector con las soluciones exactas para graficar la función exacta
for _ in range(numeroIteraciones * 100): # Multiplicamos por 100 para mejorar la precisión
    t_i = vectorSolucion_t[-1] + delta_t / 100 # Usamos el último elemento del vector
    vectorSolucion_t = np.append(vectorSolucion_t, t_i)

# Evaluamos la función en cada paso y almacenamos el resultado en la corespondiente lista vectorSolucion_x
vectorSolucion_x_1 = [solucionSimbolica_1.subs({t: valor}) for valor in list(vectorSolucion_t)]
vectorSolucion_x_2 = [solucionSimbolica_2.subs({t: valor}) for valor in list(vectorSolucion_t)]
vectorSolucion_x_3 = [solucionSimbolica_3.subs({t: valor}) for valor in list(vectorSolucion_t)]

# Definimos las soluciones simbólicas en una lista para poder iterar sobre ellas
soluciones_simbolicas = [solucionSimbolica_1, solucionSimbolica_2, solucionSimbolica_3]

# Evaluamos la función en cada paso y almacenamos el resultado en la correspondiente lista vectorSolucion_x
vectorSolucion_x = [[sol.subs({t: valor}) for valor in vectorSolucion_t] for sol in soluciones_simbolicas]

plt.figure(figsize=(8, 6))
plt.text(-5.6, 0, 'x(t)', rotation='horizontal', fontsize=12)  # Colocar 'x(t)' verticalmente
plt.xticks(np.arange(-4, 5, 1))  # Marcas cada unidad en el eje x
plt.yticks(np.arange(-3.14, 4.14, 0.5))  # Marcas cada unidad en el eje y
plt.grid(True, linewidth=0.3)  # Ajusta el ancho de las líneas de la cuadrícula
plt.axhline(0, color='black', linewidth=0.8)  # Línea horizontal en y=0
plt.axvline(0, color='black', linewidth=0.8)  # Línea vertical en x=0
plt.xlabel('$t$')
plt.plot(
    vectorSolucion_t,
    vectorSolucion_x_1,
    label="0.300",
    color='navy',
    linestyle='-',
    linewidth=0.9,
    markersize=12, 
)
plt.plot(
    vectorSolucion_t,
    vectorSolucion_x_2,
    label="0.301",
    color='orange',
    linestyle='-.',
    alpha = 0.6,
    linewidth=1.1,
    markersize=12, 
)
plt.plot(
    vectorSolucion_t,
    vectorSolucion_x_3,
    label="0.302",
    color="purple",
    linestyle='--',
    alpha = 0.4,
    linewidth=1.2,
    markersize=12, 
)

plt.legend()  # Esto coloca la leyenda en la esquina superior izquierda
plt.title("Familia de soluciones")
plt.savefig("Familia de soluciones.pdf")  # Con esta se guarda la imagen en formato pdf
plt.show()

####################################################################################################


import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

def funcion_f(ecuacion: str = "x"):
    return sp.sympify(ecuacion)

def FamiliaSoluciones(C: float):
    t = sp.symbols('t')
    return 2 * sp.atan(C * sp.exp(sp.exp(t)))

# Definimos las constantes de integración, porducto
# del depeje de la familia de soluciones por el valor inicial
C_1 = ((np.tan(3/20)) / (sp.exp(sp.exp(3/10))))
C_2 = ((np.tan(301/2000)) / (sp.exp(sp.exp(301/1000))))
C_3 = ((np.tan(151/1000)) / (sp.exp(sp.exp(151/500))))

# Calculamos las soluciones simbólicas
solucionSimbolica_1 = FamiliaSoluciones(C_1)
solucionSimbolica_2 = FamiliaSoluciones(C_2)
solucionSimbolica_3 = FamiliaSoluciones(C_3)

# Parámetros
numeroIteraciones = 8
pasoInicial_t = -4
pasoFinal_t = 4

# Calculamos el tamaño de paso
delta_t = (pasoFinal_t - pasoInicial_t) / numeroIteraciones

# Iteramos para generar el vector de valores en el eje t
vectorSolucion_t = np.linspace(pasoInicial_t, pasoFinal_t, numeroIteraciones * 100)
print(vectorSolucion_t)

# Evaluamos las soluciones en cada paso y generamos un vecotor con las tres soluciones que contienen esas evaluaciones
vectorSolucion_x = [[sol.subs({'t': valor}) for valor in vectorSolucion_t] for sol in [solucionSimbolica_1, solucionSimbolica_2, solucionSimbolica_3]]

# Configuración de la gráfica
plt.figure(figsize=(8, 6))
plt.text(-5.6, 0, 'x(t)', rotation='horizontal', fontsize=12)
plt.xticks(np.arange(-4, 5, 1))
plt.yticks(np.arange(-3.14, 4.14, 0.5))
plt.grid(True, linewidth=0.3)
plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)
plt.xlabel('$t$')

# Graficamos las soluciones
for i, vectorSolucion_x_i in enumerate(vectorSolucion_x, start=1):
    plt.plot(
        vectorSolucion_t,
        vectorSolucion_x_i,
        label=f"{0.3 + i/100:.3f}",  # Etiqueta con valor de C
        linestyle='-',  # Estilo de línea
        linewidth=1.1 + i * 0.1,  # Ancho de línea
        alpha=0.4 + i * 0.2,  # Transparencia
        color=plt.cm.viridis(i / len(vectorSolucion_x)),  # Color
        markersize=12,
    )

plt.legend()
plt.title("Familia de soluciones")
plt.savefig("Familia de soluciones.pdf")
plt.show()
