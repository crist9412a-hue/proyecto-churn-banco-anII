"""
churn_utils.py
==============
Funciones compartidas por todos los notebooks del proyecto de Churn Bancario.

La idea es NO repetir código: cada notebook hace `from churn_utils import *`
y así todos usan exactamente las mismas métricas, gráficas y rutas.

Contenido:
    1. Rutas del proyecto
    2. Métricas (las mismas funciones vistas en clase: sensibilidad, especificidad, VPP, VPN)
    3. Gráficas (curva ROC, matriz de confusión, ganancia acumulada / lift)
    4. Evaluación completa de un modelo en una sola llamada
    5. Guardar / cargar datos preparados y resultados de cada modelo
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, auc, roc_auc_score, confusion_matrix,
                             f1_score, average_precision_score, precision_recall_curve)

# ---------------------------------------------------------------------------
# 1. Rutas del proyecto (todas relativas a la carpeta donde está este archivo)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTADOS_DIR = os.path.join(BASE_DIR, "resultados")
MODELOS_DIR = os.path.join(BASE_DIR, "modelos")
FIGURAS_DIR = os.path.join(BASE_DIR, "figuras")

RUTA_DATOS_CRUDOS = os.path.join(DATA_DIR, "bank_churn.xlsx")
RUTA_DATOS_MODELO = os.path.join(DATA_DIR, "datos_modelo.pkl")

for _d in (DATA_DIR, RESULTADOS_DIR, MODELOS_DIR, FIGURAS_DIR):
    os.makedirs(_d, exist_ok=True)

SEMILLA = 77300  # misma semilla usada en clase


# ---------------------------------------------------------------------------
# 2. Métricas (Confusion Matrix devuelve: cm[0,0], cm[0,1], cm[1,0], cm[1,1] = tn, fp, fn, tp)
# ---------------------------------------------------------------------------
def _cm(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    return cm[0][0], cm[0][1], cm[1][0], cm[1][1]


def custom_sensitivity_score(y_test, y_pred):
    """Sensibilidad (Recall): de los clientes que SÍ se fueron, ¿cuántos detectamos?"""
    tn, fp, fn, tp = _cm(y_test, y_pred)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def custom_specificity_score(y_test, y_pred):
    """Especificidad: de los clientes que se quedaron, ¿cuántos clasificamos bien?"""
    tn, fp, fn, tp = _cm(y_test, y_pred)
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def custom_ppv_score(y_test, y_pred):
    """Valor Predictivo Positivo (Precision): de los que marcamos como fuga, ¿cuántos se fueron?"""
    tn, fp, fn, tp = _cm(y_test, y_pred)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def custom_npv_score(y_test, y_pred):
    """Valor Predictivo Negativo: de los que marcamos como 'se queda', ¿cuántos se quedaron?"""
    tn, fp, fn, tp = _cm(y_test, y_pred)
    return tn / (tn + fn) if (tn + fn) > 0 else 0.0


def custom_accuracy_score(y_test, y_pred):
    tn, fp, fn, tp = _cm(y_test, y_pred)
    return (tn + tp) / (tn + tp + fp + fn)


# ---------------------------------------------------------------------------
# 3. Gráficas
# ---------------------------------------------------------------------------
def plot_roc(y_test, y_pred_prob, titulo="Curva ROC"):
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob, pos_label=1, drop_intermediate=False)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5, 4.5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC (AUC = %0.3f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    plt.xlim([-0.001, 1.001]); plt.ylim([-0.001, 1.001])
    plt.xlabel("1 - Especificidad (Tasa de Falsos Positivos)")
    plt.ylabel("Sensibilidad (Tasa de Verdaderos Positivos)")
    plt.title(titulo)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def plot_confusion(y_test, y_pred, titulo="Matriz de confusión"):
    """Matriz de confusión en formato visual (filas = real, columnas = predicho)."""
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(cm, cmap="Blues")
    etiquetas = ["Se queda (0)", "Se va (1)"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(etiquetas)
    ax.set_yticks([0, 1]); ax.set_yticklabels(etiquetas)
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real"); ax.set_title(titulo)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    plt.tight_layout()
    plt.show()


def tabla_ganancia(y_test, y_pred_prob, n_grupos=10):
    """
    Tabla de ganancia acumulada (gains) y lift por deciles.
    Ordena los clientes de mayor a menor probabilidad de fuga y responde:
    'Si contacto al X% de clientes con mayor riesgo, ¿qué % de las fugas capturo?'
    """
    d = pd.DataFrame({"y": np.asarray(y_test), "p": np.asarray(y_pred_prob)})
    d = d.sort_values("p", ascending=False).reset_index(drop=True)
    d["decil"] = pd.qcut(d.index, n_grupos, labels=range(1, n_grupos + 1))
    t = d.groupby("decil", observed=True).agg(clientes=("y", "size"), fugas=("y", "sum"),
                                              prob_min=("p", "min"), prob_max=("p", "max"))
    t["tasa_fuga"] = t["fugas"] / t["clientes"]
    t["%_clientes_acum"] = t["clientes"].cumsum() / t["clientes"].sum()
    t["%_fugas_capturadas_acum"] = t["fugas"].cumsum() / t["fugas"].sum()
    t["lift"] = t["tasa_fuga"] / d["y"].mean()
    t["lift_acum"] = t["%_fugas_capturadas_acum"] / t["%_clientes_acum"]
    return t


# ---------------------------------------------------------------------------
# 4. Evaluación completa de un modelo
# ---------------------------------------------------------------------------
def evaluar_modelo(nombre, y_test, y_pred_prob, class_threshold=0.5, graficar=True):
    """
    Imprime y devuelve todas las métricas del clasificador (mismo formato visto en clase).
    y_pred_prob: probabilidad de la clase positiva (1 = el cliente se va).
    """
    y_test = np.asarray(y_test)
    y_pred = np.where(np.asarray(y_pred_prob) > class_threshold, 1, 0)

    metricas = {
        "Modelo": nombre,
        "Umbral": class_threshold,
        "AUC": roc_auc_score(y_test, y_pred_prob),
        "PR-AUC": average_precision_score(y_test, y_pred_prob),
        "Accuracy": custom_accuracy_score(y_test, y_pred),
        "Sensibilidad": custom_sensitivity_score(y_test, y_pred),
        "Especificidad": custom_specificity_score(y_test, y_pred),
        "Precision (VPP)": custom_ppv_score(y_test, y_pred),
        "VPN": custom_npv_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
    }

    print(f"Métricas del modelo: {nombre}  (umbral = {class_threshold})\n")
    cm = np.transpose(confusion_matrix(y_test, y_pred, labels=[0, 1]))
    print("Matriz de confusión (formato clase: filas = predicho, columnas = real):\n" + str(cm))
    for k in ["Accuracy", "Sensibilidad", "Especificidad", "Precision (VPP)", "VPN", "F1", "AUC", "PR-AUC"]:
        print(f" {k:<16}: {metricas[k]:.4f}")

    if graficar:
        plot_roc(y_test, y_pred_prob, titulo=f"Curva ROC - {nombre}")
    return metricas


def umbral_optimo_f1(y_true, y_prob):
    """Umbral que maximiza F1 (balance entre detectar fugas y no gastar beneficios de más)."""
    p, r, t = precision_recall_curve(y_true, y_prob)
    f1 = 2 * p * r / (p + r + 1e-12)
    i = np.nanargmax(f1[:-1])
    return float(t[i]), float(f1[i])


# ---------------------------------------------------------------------------
# 5. Guardar / cargar datos y resultados
# ---------------------------------------------------------------------------
def cargar_datos_modelo():
    """Carga los datos que deja listos el notebook 00 (train/test ya separados)."""
    if not os.path.exists(RUTA_DATOS_MODELO):
        raise FileNotFoundError("Primero ejecute el notebook 00_EDA_Preprocesamiento.ipynb")
    with open(RUTA_DATOS_MODELO, "rb") as f:
        return pickle.load(f)


def guardar_resultados(nombre, y_test, y_pred_prob, metricas=None, mejores_parametros=None, extra=None):
    """Guarda las probabilidades de test y métricas de un modelo para el notebook de comparación."""
    res = {"nombre": nombre, "y_test": np.asarray(y_test), "y_prob": np.asarray(y_pred_prob),
           "metricas": metricas, "mejores_parametros": mejores_parametros, "extra": extra}
    archivo = os.path.join(RESULTADOS_DIR, nombre.replace(" ", "_").replace("/", "-") + ".pkl")
    with open(archivo, "wb") as f:
        pickle.dump(res, f)
    print(f"Resultados guardados en: {os.path.relpath(archivo, BASE_DIR)}")


def cargar_resultados():
    """Lee todos los resultados guardados en /resultados y devuelve un diccionario {nombre: resultado}."""
    out = {}
    for f in sorted(os.listdir(RESULTADOS_DIR)):
        if f.endswith(".pkl"):
            with open(os.path.join(RESULTADOS_DIR, f), "rb") as fh:
                r = pickle.load(fh)
            out[r["nombre"]] = r
    return out
