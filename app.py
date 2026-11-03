import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# --- Configuración de la página ---
st.set_page_config(page_title="Gradiente Descendente", layout="centered")
st.title("Visualización del Gradiente Descendente")
st.write("Minimizando la función $f(x) = x^2$")

# --- Funciones matemáticas ---
def f(x):
    return x**2

def gradient(x):
    return 2 * x

def gradient_descent(x_start, alpha, num_iterations):
    x = x_start
    trayectory = [x]
    for _ in range(num_iterations):
        x = x - alpha * gradient(x)
        trayectory.append(x)
    return np.array(trayectory)

# --- Interfaz de usuario (Sidebar) ---
st.sidebar.header("Parámetros del Gradiente")
x_start = st.sidebar.slider("Punto de inicio (x_start)", -6.0, 6.0, 5.0)
alpha = st.sidebar.slider("Tasa de aprendizaje (alpha)", 0.01, 1.05, 0.9, 0.01)
num_iterations = st.sidebar.slider("Iteraciones", 1, 50, 20)

# --- Cálculo ---
trayectory = gradient_descent(x_start, alpha, num_iterations)

# --- Gráfica ---
x = np.linspace(-6, 6, 400)
y = f(x)

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(x, y, label='$f(x)=x^2$', color='blue')
ax.scatter(trayectory, f(trayectory), color='red', zorder=5)
ax.plot(trayectory, f(trayectory), color='red', linestyle="--", label='Trayectoria')
ax.set_xlabel('$x$')
ax.set_ylabel('$f(x)$')
ax.set_title(f'Gradiente Descendente ($\\alpha={alpha}$)')
ax.legend()
ax.grid(True)

# Mostrar la figura en Streamlit en lugar de plt.show()
st.pyplot(fig)
