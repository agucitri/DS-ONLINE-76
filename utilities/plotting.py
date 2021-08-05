import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go


def barplot(dataframe: pd.DataFrame, column: str, **kwargs):
    """
    Grafico de barras sobre una columna de interes
    :param dataframe: DataFrame
    :param column: columna de interes
    :return: NoneType
    """
    data = dataframe[column].value_counts().sort_values(ascending=True)
    bars = tuple(data.index.tolist())
    values = data.values.tolist()
    y_pos = np.arange(len(bars))
    colors = ["lightblue"] * len(bars)
    colors[-1] = "blue"
    plt.figure(figsize=(16, 10), **kwargs)
    plt.barh(y_pos, values, color=colors)
    plt.title(f"Distribución de {column}")
    plt.yticks(ticks=y_pos, labels=bars)
    return plt.show()


def box_plot(dataframe: pd.DataFrame, column: str, label: str = None, **kwargs):
    """
    Diagrama de caja y bigotes para ver la distribucion de una variable de interes y de esa variable con respeto
    a una variable categorica
    :param dataframe: DataFrame
    :param column: variable de interes
    :param label: variable categorica (suele ser la variable dependiente si es un problema de clasificacion)
    :return: NoneType
    """
    if label is None:
        return sns.catplot(data=dataframe, x=column, kind="box", **kwargs)
    else:
        return sns.catplot(data=dataframe, x=label, y=column, kind="box", **kwargs)


def distribution_plot(dataframe: pd.DataFrame, column: str, **kwargs):
    """
    Histograma de una columna de interes
    :param dataframe: DataFrame
    :param column: columna de interes
    :return: Nonetype
    """
    return sns.displot(data=dataframe, x=column, **kwargs)


def heat_map(dataframe: pd.DataFrame, **kwargs):
    """
    Matriz de calor sobre un DataFrame
    :param dataframe: DataFrame
    :return:
    """
    return sns.heatmap(dataframe, annot=True, fmt=".1g", linecolor="w", linewidths=3, **kwargs)


def radar_chart(df: pd.DataFrame, columns: list, label: str, title: str):
    """
    Radar chart filtering by the categories of the target variable
    :param title: title
    :param df: DataFrame
    :param columns: columns of interest
    :param label: target variable
    :return:
    """
    categories = sorted(df[label].unique().tolist())
    table = np.round(df.groupby(by=label, as_index=True)[columns].mean(), decimals=2)
    figure = go.Figure()
    for categorie in categories:
        figure.add_trace(go.Scatterpolar(r=table.loc[categorie].tolist(),
                                         theta=columns,
                                         fill='toself',
                                         name=categorie))
    figure.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                         showlegend=True, title=title)
    return figure.show()