# Proyecto Final – Predicción de Churn en una entidad financiera
Métodos de Analítica II 

## Cómo abrirlo en Visual Studio Code
1. Descargue el repositorio y ábralo completa en VS Code (`Archivo > Abrir carpeta…`).
2. Instale las extensiones **Python** y **Jupyter** de Microsoft.
3. Instale las librerías (una sola vez), en la terminal de VS Code:
   ```
   pip install -r requirements.txt
   ```
4. Abra cualquier notebook y elija el kernel de Python donde instaló las librerías.

### Entorno recomendado 

Para correr el entorno se utilizo python 3.12 Con el administrador de Python de python.org:
```
py install 3.12
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
En VS Code elija el kernel `.venv` al abrir cada notebook.



## Orden de ejecución
| # | Notebook | Qué hace | Tiempo aprox. |
|---|---|---|---|
| 00 | `00_EDA_Preprocesamiento.ipynb` | EDA (pruebas estadísticas, Information Value, interacciones), variables nuevas, dummies, partición 80/20 → `data/datos_modelo.pkl` | 1 min |
| 01 | `01_Metodo_Gradiente.ipynb` | Regresión logística con el método del gradiente a mano (batch, mini-batch, SGD, L1/L2) | 1 min |
| 02 | `02_SVM_Kernels_y_SVR.ipynb` | SVM: kernels lineal, poly, RBF, sigmoide + SVR | 4 min |
| 03 | `03_Redes_Neuronales.ipynb` | Redes neuronales Keras (scikeras) + MLP sklearn | 8 min |
| 04 | `04_Gradient_Boosting.ipynb` | Gradient Boosting a mano, GBM, HistGradientBoosting | 5 min |
| 05 | `05_Ensambles.ipynb` | Random Forest, Bagging, Extra Trees, AdaBoost, Voting, Stacking | 5 min |
| 06 | `06_Comparacion_y_Seleccion.ipynb` | Comparación, umbral de negocio, lift, modelo final, lista priorizada | 1 min |
| ★ | `Proyecto_Final_General.ipynb` | **Notebook principal: ejecuta los notebooks 00 a 06 con `%run` y muestra el resumen** | 25 min |

El orden sigue los temas del curso: optimización (método del gradiente), SVM, redes neuronales y ensambles (boosting y bagging).
El notebook **00 debe ejecutarse primero**. Del 01 al 04 son independientes entre sí; el 05 (Voting/Stacking) usa los mejores
hiperparámetros del 02 y el 04 si ya existen, y el 06 necesita los resultados de todos.

## Cómo están conectados (diseño modular)
`Proyecto_Final_General.ipynb` es el **orquestador**: no repite código, ejecuta cada notebook con `%run`.
Los notebooks se comunican por archivos y comparten las funciones de `churn_utils.py`.
```
Proyecto_Final_General.ipynb
   ├── %run 00_EDA ──► data/datos_modelo.pkl (mismo train/test para todos)
   ├── %run 01_Gradiente ─┐
   ├── %run 02_SVM       ─┤
   ├── %run 03_Redes     ─┤──► resultados/*.pkl (probabilidades y métricas de cada modelo)
   ├── %run 04_GBM       ─┤
   ├── %run 05_Ensambles ─┘
   └── %run 06_Comparacion ──► modelos/modelo_final_churn.joblib
                            └► resultados/lista_priorizada_clientes.xlsx
churn_utils.py ← importado por todos (métricas de clase, ROC, lift, guardar/cargar)
```
Si se modifica un notebook individual, el principal usa automáticamente la versión nueva.

## Carpetas
- `data/` – base original (`bank_churn.xlsx`) y datos preparados.
- `resultados/` – probabilidades y métricas de cada modelo, tabla comparativa y lista priorizada de clientes.
- `modelos/` – modelo final guardado con joblib.
- `figuras/` – gráficas para el informe.
