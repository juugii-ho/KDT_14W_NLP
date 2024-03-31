import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV



def explore_data(df, columns=None):
    """
    데이터를 탐색하고 시각화합니다.

    Args:
        df: 탐색할 데이터

    Returns:
        None
    """
    print(f'데이터 정보 출력 : {df.info()}')
    print(f'통계 요약 : {df.describe()}')
    print(f'결측치 파악 : {df.isnull().sum()}')

    # 히스토그램
    plt.hist(df)
    # 상관관계 히트맵
    sns.heatmap(df.corr())
    # 이상치 박스플롯
    if columns is None:
        columns = df.columns
    fig, axes = plt.subplots(nrows=1, ncols=len(columns), figsize=(15, 5))
    for i, column in enumerate(columns):
        sns.boxplot(x=df[column], ax=axes[i])
        axes[i].set_title(column)
    plt.tight_layout()
    plt.show()


pd.DataFrame([[1,2,3,4],[1,2,3,4]])