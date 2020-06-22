#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[133]:


import pandas as pd
import numpy as np


# In[134]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[135]:


black_friday.head(5)


# In[136]:


black_friday.shape


# In[169]:


soma = ((black_friday['Gender'] == 'F') & (black_friday['Age']=='26-35')).sum()


# In[170]:


soma


# In[ ]:





# In[139]:


usuarios_unicos = black_friday['User_ID'].nunique()


# In[140]:


usuarios_unicos


# In[141]:


tipos_dados = black_friday.dtypes.nunique()


# In[142]:


tipos_dados


# In[143]:


percentual_faltante = black_friday.isna().sum().max() / black_friday.shape[0]


# In[144]:


percentual_faltante


# In[145]:


black_friday.isnull().sum()


# In[146]:


valores_nulos = black_friday['Product_Category_3'].isnull().sum()


# In[147]:


valores_nulos


# In[148]:


valor_mais_frequente = black_friday['Product_Category_3'].mode()


# In[149]:


float(valor_mais_frequente) 


# In[150]:


from sklearn import preprocessing
# Create x, where x the 'scores' column's values as floats
x = black_friday[['Purchase']].values.astype(float)

# Create a minimum and maximum processor object
min_max_scaler = preprocessing.MinMaxScaler()

# Create an object to transform the data to fit minmax processor
x_scaled = min_max_scaler.fit_transform(x)

# Run the normalizer on the dataframe
normalized = pd.DataFrame(x_scaled)


# In[151]:


float(normalized.mean())


# In[152]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc_normalized = sc.fit_transform(normalized)


# In[153]:


sc_normalized


# In[154]:


int(((sc_normalized >= -1) & (sc_normalized <= 1)).sum())


# In[155]:


dados_observados = black_friday[black_friday['Product_Category_2'].isnull()]


# In[156]:


dados_observados


# In[157]:


bool(dados_observados['Product_Category_2'].sum() == dados_observados['Product_Category_3'].sum())


# In[ ]:





# In[ ]:





# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[158]:


def q1():
    # Retorne aqui o resultado da questão 2.
    return black_friday.shape
    pass


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[168]:


def q2():
    # Retorne aqui o resultado da questão 2.
    return int(((black_friday['Gender'] == 'F') & (black_friday['Age'] == '26-35')).sum())
    pass


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[160]:


def q3():
    # Retorne aqui o resultado da questão 3.
    usuarios_unicos = black_friday['User_ID'].nunique()
    return usuarios_unicos
    pass


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[161]:


def q4():
    # Retorne aqui o resultado da questão 4.
    tipos_dados = black_friday.dtypes.nunique()
    return tipos_dados
    pass


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[162]:


def q5():
    # Retorne aqui o resultado da questão 5.
    percentual_faltante = black_friday.isna().sum().max() / black_friday.shape[0]
    return percentual_faltante
    pass


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[163]:


def q6():
    # Retorne aqui o resultado da questão 6.
    valores_nulos = black_friday['Product_Category_3'].isnull().sum()
    return int(valores_nulos)
    pass


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[164]:


def q7():
    # Retorne aqui o resultado da questão 7.
    valor_mais_frequente = black_friday['Product_Category_3'].mode()
    return float(valor_mais_frequente)
    pass


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[165]:


def q8():
    # Retorne aqui o resultado da questão 8.
    from sklearn import preprocessing
    # Create x, where x the 'scores' column's values as floats
    x = black_friday[['Purchase']].values.astype(float)
    # Create a minimum and maximum processor object
    min_max_scaler = preprocessing.MinMaxScaler()
    # Create an object to transform the data to fit minmax processor 
    x_scaled = min_max_scaler.fit_transform(x)
    # Run the normalizer on the dataframe
    normalized = pd.DataFrame(x_scaled)
    return float(normalized.mean())
    pass


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[166]:


def q9():
    # Retorne aqui o resultado da questão 9.
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    sc_normalized = sc.fit_transform(normalized)
    return int(((sc_normalized >= -1) & (sc_normalized <= 1)).sum())
    pass


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[167]:


def q10():
    # Retorne aqui o resultado da questão 10.
    dados_observados = black_friday[black_friday['Product_Category_2'].isnull()]
    return bool(dados_observados['Product_Category_2'].sum() == dados_observados['Product_Category_3'].sum())
    pass


# In[ ]:




