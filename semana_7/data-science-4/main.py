#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[18]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk


# In[19]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

#from IPython.core.pylabtools import figsize


#figsize(12, 8)

#sns.set()


# In[20]:


countries = pd.read_csv("countries.csv")


# In[21]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[22]:


# Sua análise começa aqui.
countries.info()


# In[23]:


#Selecionando colunas com vírgula
nomes_colunas = countries.columns.drop(['Country','Region','Population','Area','GDP'])
nomes_colunas = list(nomes_colunas)
nomes_colunas


# In[24]:


# Modificando nas colunas selecionadas vírgula para ponto e transformando-as em float  
countries[nomes_colunas] = countries[nomes_colunas].apply(lambda x: x.str.replace(',', '.').astype('float'))
countries


# In[25]:


# Removendo espaços nas colunas Country e Region
countries['Country'] = countries['Country'].str.strip()
countries['Region'] = countries['Region'].str.strip()
countries


# In[26]:


#Informação das colunas 
countries.info()


# # Análise de Dados : Questão 1

# In[27]:


# Selecionando as Regiões únicas em ordem alfabética
regioes = countries['Region'].sort_values().unique()
regioes = list(regioes)
regioes


# # Análise de Dados: Questão 2

# In[ ]:


from sklearn.preprocessing import KBinsDiscretizer


# In[31]:


# Aplicando o KBinsDiscretizer
discretizer = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")
discretizer.fit(countries[['Pop_density']])
# Obtendo um array com os dados transformados
score_bins = discretizer.transform(countries[["Pop_density"]])
score_bins


# In[32]:


# Encontrando o percentil 90
q_90= np.quantile(score_bins, 0.9)
q_90


# In[33]:


#Contando quantos valores estão acima do percentil 90
count = 0
for x in score_bins:
    if (x > q_90):
        count = count + 1
count    


# # Análise de Dados: Questão 3

# In[39]:


# Selecionando as colunas Region e Climate dos dados e criando um novo dataframe apenas com essas 2 colunas
region = countries['Region']
climate = countries['Climate']
new_dataframe=pd.concat([region, climate], axis=1)
new_dataframe


# In[40]:


# Transformando o valor nan em um valor médio dos dados na coluna Climate
new_dataframe = new_dataframe.fillna(new_dataframe.mean())


# In[41]:


# Aplicando o one-hot encoding  usando a funçao get_dummies do Pandas
encoded_columns = pd.get_dummies(data=new_dataframe, columns=['Region', 'Climate'])
encoded_columns


# In[42]:


# Contando quantos novos atributos(colunas novas) foram criados
length_encoded = len(list(encoded_columns))
length_encoded


# # Análise de Dados: Questão 4

# In[43]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


# In[45]:


#Criando a pipeline
pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("standard_scaler", StandardScaler())
    ])
pipeline


# In[47]:


# Criando um novo dataframe apenas com valores numéricos
df_numeric = countries.select_dtypes(include=['float64','int64'])
df_numeric


# In[48]:


# Fazendo o ajuste dos dados do novo dataframe no pipeline 
pipeline_transf = pipeline.fit_transform(df_numeric)
pipeline_transf


# In[56]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[57]:


# Usando o pipeline no teste e encontrando o valor correspondente à variável Arable
result = pipeline.transform([test_country[2:]])
result[0][9]


# In[58]:


result = float(result[0][9].round(3))


# # Análise de Dados:Questão 5

# In[60]:


# Criando um novo dataframe da coluna Net Migration do dataframe 
net_migration = countries['Net_migration']
net_migration


# In[62]:


# Visualisando os dados da coluna selecionada num bloxpot
import matplotlib.pyplot as plt
plt.figure(figsize=(12,8))
sns.boxplot(net_migration, orient="vertical");


# In[63]:


# Aplicando o método do boxplot para encontrar a faixa dos dados considerada normal
q1 = net_migration.quantile(0.25)
q3 = net_migration.quantile(0.75)
iqr = q3 - q1

non_outlier_interval_iqr = [q1 - 1.5 * iqr, q3 + 1.5 * iqr]

print(f"Faixa considerada \"normal\": {non_outlier_interval_iqr}")


# In[64]:


# Descobrindo o número de outliers da variável
outliers_iqr = net_migration[(net_migration < non_outlier_interval_iqr[0]) | (net_migration > non_outlier_interval_iqr[1])]
len(outliers_iqr)


# In[65]:


# Encontrando o número de outliers abaixo
outliers_abaixo = net_migration[(net_migration < non_outlier_interval_iqr[0])]
outliers_abaixo = len(outliers_abaixo)
outliers_abaixo


# In[66]:


# Encontrando o número de outliers acima
outliers_acima = net_migration[(net_migration > non_outlier_interval_iqr[1])]
outliers_acima = len(outliers_acima)
outliers_acima


# In[68]:


# Pela análise gráfica e quantitativa do bloxpot, vemos que há uma grande quantidade de outliers,
# então não deveríamos remover esses pontos por esse método , resposta False
len(net_migration)


# # Análise de Dados: Questão 6

# In[71]:


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer


# In[72]:


# Carregando as seguintes categorias e o dataset newsgroups
categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)


# In[73]:


# Aplicando o CountVectorizer
vec = CountVectorizer()
X = vec.fit_transform(newsgroup['data'])
X


# In[74]:


# Criando um novo dataframe com todas as possíveis palavras
import pandas as pd
phone = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
phone


# In[75]:


# Contando quantas vezes aparece a palavra phone 
num_phone = phone['phone'].sum()
num_phone = int(num_phone)
num_phone


# # Análise de Dados: Questão 7

# In[76]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[77]:


#Aplicando o TfidfVectorizer ao data set newsgroups
vec = TfidfVectorizer()
X = vec.fit_transform(newsgroup['data'])


# In[78]:


# Criando um novo dataframe com todas as possíveis palavras
tfid_phone = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
tfid_phone


# In[79]:


# Encontrando o TF-IDF da palavra phone
result_tfid = tfid_phone['phone'].sum()
result_tfid = float(result_tfid.round(3))
result_tfid


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[51]:


def q1():
    # Retorne aqui o resultado da questão 1.
    return regioes


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[7]:


def q2():
    # Retorne aqui o resultado da questão 2.
    return count


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[109]:


def q3():
    # Retorne aqui o resultado da questão 3.
    return length_encoded


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[134]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[128]:


def q4():
    # Retorne aqui o resultado da questão 4.
    return result


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[12]:


def q5():
    # Retorne aqui o resultado da questão 4.
    return outliers_abaixo, outliers_acima, False


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[9]:


def q6():
    # Retorne aqui o resultado da questão 4.
    return num_phone


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[22]:


def q7():
    # Retorne aqui o resultado da questão 4.
    return result_tfid

