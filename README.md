# Calibración Econométrica de Series Financieras: Modelos AR-GARCH y Hechos Estilizados (IBEX 35)

Estudio teórico y computacional en Python sobre la calibración de procesos autorregresivos y de volatilidad condicional estocástica, aplicado tanto a series simuladas como a los **log-retornos históricos del índice IBEX 35**.

## 🔍 Aspectos Clave del Proyecto
* **Equivalencia Matemática y Estimación:** Demostración analítica e implementación computacional de la equivalencia entre Mínimos Cuadrados (LS) y Máxima Verosimilitud (MLE) bajo ruido Gaussiano, extendiendo el marco MLE a distribuciones leptocúrticas (**t-Student**).
* **Hechos Estilizados de Retornos Financieros (Rama Cont):** Análisis empírico del IBEX 35 demostrando ausencia de autocorrelación lineal en los retornos, presencia de **colas pesadas** (curtosis de $5,28$ / exceso de curtosis de $2,28$), asimetría negativa y fuerte **agrupamiento de volatilidad (*volatility clustering*)** visible en la autocorrelación de los retornos absolutos y cuadráticos.
* **Optimización Numérica No Lineal (`scipy.optimize`):** Calibración de 6 especificaciones incrementales (desde ruido Gaussiano/t-Student puro hasta **AR(1) + GARCH(1,1) con innovaciones t-Student**) comparando algoritmos libres de gradiente (`Nelder-Mead`), cuasi-Newton (`BFGS`) y optimizadores restringidos (`SLSQP`, `L-BFGS-B`).
* **Restricciones Estructurales y Diagnóstico de Residuos:** Imposición algorítmica de estacionariedad en media, positividad de varianza condicional, estacionariedad en covarianza y existencia de varianza finita. Validación mediante estandarización de innovaciones, correlogramas ACF y QQ-Plots.
