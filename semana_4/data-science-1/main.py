#!/usr/bin/env python
# coding: utf-8

# # Desafio 3
# 
# Neste desafio, iremos praticar nossos conhecimentos sobre distribuições de probabilidade. Para isso,
# dividiremos este desafio em duas partes:
#     
# 1. A primeira parte contará com 3 questões sobre um *data set* artificial com dados de uma amostra normal e
#     uma binomial.
# 2. A segunda parte será sobre a análise da distribuição de uma variável do _data set_ [Pulsar Star](https://archive.ics.uci.edu/ml/datasets/HTRU2), contendo 2 questões.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF


# In[5]:


#%matplotlib inline

#from IPython.core.pylabtools import figsize


#figsize(12, 8)

#sns.set()


# ## Parte 1

# ### _Setup_ da parte 1

# In[6]:


np.random.seed(42)
    
dataframe = pd.DataFrame({"normal": sct.norm.rvs(20, 4, size=10000),
                     "binomial": sct.binom.rvs(100, 0.2, size=10000)})


# ## Inicie sua análise a partir da parte 1 a partir daqui

# In[7]:


# Sua análise da parte 1 começa aqui.
dataframe
df_normal = dataframe['normal']
df_normal
df_bin = dataframe['binomial']
df_bin


# In[8]:


df_normal_sort = df_normal.sort_values()
df_normal_sort = df_normal_sort.reset_index(drop=True)
df_bin_sort = df_bin.sort_values()
df_bin_sort = df_bin_sort.reset_index(drop=True)
df_normal_sort


# In[9]:


normal_q = df_normal_sort.quantile((0.25, 0.5, 0.75))
bin_q = df_bin_sort.quantile((0.25,0.5,0.75))


# In[10]:


q_norm = list(normal_q)
q_bin = list(bin_q)
iqr = (normal_q-bin_q).round(3)
iqr_tuple =tuple(iqr)


# In[11]:


iqr_tuple


# In[12]:


mean_norm = df_normal.mean()
std_norm = df_normal.std()
var_norm = df_normal.var()


# In[13]:


mean_norm
std_norm


# In[14]:


lim_inf = mean_norm - std_norm
lim_sup = mean_norm + std_norm 


# In[15]:


cdf_norm = ECDF(df_normal)
prob = round(cdf_norm(lim_sup) - cdf_norm(lim_inf),3)
float(prob)


# In[16]:


mean_bin = df_bin.mean()
std_bin = df_bin.std()
var_bin = df_bin.var()


# In[17]:


diff = float(mean_bin-mean_norm)
diff = round(diff,3)
diff_2 = float(var_bin - var_norm)
diff_2 = round(diff_2,3)


# ## Questão 1
# 
# Qual a diferença entre os quartis (Q1, Q2 e Q3) das variáveis `normal` e `binomial` de `dataframe`? Responda como uma tupla de três elementos arredondados para três casas decimais.
# 
# Em outra palavras, sejam `q1_norm`, `q2_norm` e `q3_norm` os quantis da variável `normal` e `q1_binom`, `q2_binom` e `q3_binom` os quantis da variável `binom`, qual a diferença `(q1_norm - q1 binom, q2_norm - q2_binom, q3_norm - q3_binom)`?

# In[18]:


def q1():
    # Retorne aqui o resultado da questão 1.
    df_normal = dataframe['normal']
    df_bin = dataframe['binomial']
    df_normal_sort = df_normal.sort_values()
    df_normal_sort = df_normal_sort.reset_index(drop=True)
    df_bin_sort = df_bin.sort_values()
    df_bin_sort = df_bin_sort.reset_index(drop=True)
    q_norm = list(normal_q)
    q_bin = list(bin_q)
    iqr = (normal_q-bin_q).round(3)
    iqr_tuple =tuple(iqr)
    return iqr_tuple
    pass


# Para refletir:
# 
# * Você esperava valores dessa magnitude?
# 
# * Você é capaz de explicar como distribuições aparentemente tão diferentes (discreta e contínua, por exemplo) conseguem dar esses valores?

# ## Questão 2
# 
# Considere o intervalo $[\bar{x} - s, \bar{x} + s]$, onde $\bar{x}$ é a média amostral e $s$ é o desvio padrão. Qual a probabilidade nesse intervalo, calculada pela função de distribuição acumulada empírica (CDF empírica) da variável `normal`? Responda como uma único escalar arredondado para três casas decimais.

# In[19]:


def q2():
    # Retorne aqui o resultado da questão 2.
    mean_norm = df_normal.mean()
    std_norm = df_normal.std()
    lim_inf = mean_norm - std_norm 
    lim_sup = mean_norm + std_norm 
    cdf_norm = ECDF(df_normal)
    prob = round(cdf_norm(lim_sup) - cdf_norm(lim_inf),3)
    return float(prob)
    pass


# Para refletir:
# 
# * Esse valor se aproxima do esperado teórico?
# * Experimente também para os intervalos $[\bar{x} - 2s, \bar{x} + 2s]$ e $[\bar{x} - 3s, \bar{x} + 3s]$.

# ## Questão 3
# 
# Qual é a diferença entre as médias e as variâncias das variáveis `binomial` e `normal`? Responda como uma tupla de dois elementos arredondados para três casas decimais.
# 
# Em outras palavras, sejam `m_binom` e `v_binom` a média e a variância da variável `binomial`, e `m_norm` e `v_norm` a média e a variância da variável `normal`. Quais as diferenças `(m_binom - m_norm, v_binom - v_norm)`?

# In[20]:


def q3():
    # Retorne aqui o resultado da questão 3.
    mean_bin = df_bin.mean()
    mean_norm = df_normal.mean()
    diff = float(mean_bin-mean_norm)
    diff = round(diff,3)
    return diff, diff_2
    pass


# Para refletir:
# 
# * Você esperava valore dessa magnitude?
# * Qual o efeito de aumentar ou diminuir $n$ (atualmente 100) na distribuição da variável `binomial`?

# ## Parte 2

# ### _Setup_ da parte 2

# In[21]:


stars = pd.read_csv("pulsar_stars.csv")

stars.rename({old_name: new_name
              for (old_name, new_name)
              in zip(stars.columns,
                     ["mean_profile", "sd_profile", "kurt_profile", "skew_profile", "mean_curve", "sd_curve", "kurt_curve", "skew_curve", "target"])
             },
             axis=1, inplace=True)

stars.loc[:, "target"] = stars.target.astype(bool)


# ## Inicie sua análise da parte 2 a partir daqui

# In[44]:


# Sua análise da parte 2 começa aqui.
from sklearn.preprocessing import StandardScaler
stars
var = stars[stars["target"]== 0]["mean_profile"]
var


# In[50]:


false_pulsar_mean_profile_standardized =(var - var.mean())/var.std()
false_pulsar_mean_profile_standardized
ecdf_f = ECDF(false_pulsar_mean_profile_standardized)


# In[83]:


theor_quant = [sct.norm.ppf(x) for x in [0.8, 0.9, 0.95]]

prob = tuple(ecdf_f(theor_quant).round(3))
prob
theor_quant = tuple(theor_quant)
(prob[0]-theor_quant[0]).round(3)


# In[97]:


theor_quant_2 = [sct.norm.ppf(x) for x in [0.25, 0.5, 0.75]]
theor_quant_2


# In[109]:


np.quantile(false_pulsar_mean_profile_standardized, .75)


# In[115]:


(np.quantile(false_pulsar_mean_profile_standardized, .75) - theor_quant_2[2]).round(3)


# ## Questão 4
# 
# Considerando a variável `mean_profile` de `stars`:
# 
# 1. Filtre apenas os valores de `mean_profile` onde `target == 0` (ou seja, onde a estrela não é um pulsar).
# 2. Padronize a variável `mean_profile` filtrada anteriormente para ter média 0 e variância 1.
# 
# Chamaremos a variável resultante de `false_pulsar_mean_profile_standardized`.
# 
# Encontre os quantis teóricos para uma distribuição normal de média 0 e variância 1 para 0.80, 0.90 e 0.95 através da função `norm.ppf()` disponível em `scipy.stats`.
# 
# Quais as probabilidade associadas a esses quantis utilizando a CDF empírica da variável `false_pulsar_mean_profile_standardized`? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[81]:


def q4():
    # Retorne aqui o resultado da questão 4.
    var = stars[stars["target"]== 0]["mean_profile"]
    false_pulsar_mean_profile_standardized =(var - var.mean())/var.std()
    ecdf_f = ECDF(false_pulsar_mean_profile_standardized)
    theor_quant = [sct.norm.ppf(x) for x in [0.8, 0.9, 0.95]]
    prob = tuple(ecdf_f(theor_quant).round(3))
    return prob
    pass


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?

# ## Questão 5
# 
# Qual a diferença entre os quantis Q1, Q2 e Q3 de `false_pulsar_mean_profile_standardized` e os mesmos quantis teóricos de uma distribuição normal de média 0 e variância 1? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[116]:


def q5():
    # Retorne aqui o resultado da questão 5.
    theor_quant_2 = [sct.norm.ppf(x) for x in [0.25, 0.5, 0.75]]
    
    return (np.quantile(false_pulsar_mean_profile_standardized, .25) - theor_quant_2[0]).round(3), (np.quantile(false_pulsar_mean_profile_standardized, .50) - theor_quant_2[1]).round(3), (np.quantile(false_pulsar_mean_profile_standardized, .75) - theor_quant_2[2]).round(3)
    pass


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?
# * Curiosidade: alguns testes de hipóteses sobre normalidade dos dados utilizam essa mesma abordagem.

# In[ ]:




