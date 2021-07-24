import pandas as pd
from scipy.stats import shapiro, normaltest, anderson


def normal_test(data: pd.DataFrame, column: str, alpha: float = 0.05, method: str = "shapiro"):
    """
    Calcula la prueba de normalidad sobre una variable de interes.
    :param data: DataFrame
    :param column: variable para probar normalidad
    :param alpha: nivel de significancia
    :param method: tipo de prueba de normalidad
    :return: NoneType
    """
    if method in ["shapiro", "d'agostino", "anderson"]:
        if method == "shapiro":
            stat, p = shapiro(data[column])
            print(f"Para {column} con la prueba de Shapiro-Wilks: \n")
        elif method == "d'agostino":
            stat, p = normaltest(data[column])
            print(f"Para {column} con la prueba de K^2 de D'Agostino: \n")
        else:
            result = anderson(data[column])
            print(f"Para {column} con la prueba de Anderson-Darling: \n")
            print(f"Estadistico: {round(result.statistic)}")
            alphas = [0.15, 0.1, 0.05, 0.025, 0.01]
            if alpha in alphas:
                idx = alphas.index(alpha)
                sl, cv = result.significance_level[idx], result.critical_values[idx]
                if result.statistic < result.critical_values[idx]:
                    print(f"Para un nivel de significancia: {sl} y un valor critico de: {cv} \n"
                          f"La muestra parece Gaussiana (No se rechaza H0).")
                else:
                    print(f"Para un nivel de significancia: {sl} y un valor critico de: {cv} \n"
                          f"La muestra no parece Gaussiana (Se rechaza H0).")
            else:
                ValueError("El nivel de significancia proporcionado no esta disponible para esta prueba.")
        if method in ["shapiro", "d'agostino"]:
            print(f"Estadístico: {round(stat, 3)} \n"
                  f"P-value: {round(p, 4)} \n")
            if p > alpha:
                print("La muestra parece Gaussiana (No se rechaza H0).")
            else:
                print("La muestra no parece Gaussiana (Se rechaza H0).")
    else:
        ValueError("El metodo debe ser shapiro, d'agostino o anderson")
    return None
