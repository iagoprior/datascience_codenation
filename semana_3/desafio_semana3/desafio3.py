import streamlit as st
import pandas as pd
import altair as alt

def main():
    st.image('mapa_brasil.jpg', width=400)
    # Add a selectbox to the sidebar:
    add_selectbox = st.sidebar.selectbox(
        'How would you like to be contacted?',
        ('Email', 'Home phone', 'Mobile phone')
    )

    # Add a slider to the sidebar:
    add_slider = st.sidebar.slider(
        'Select a range of values',
        0.0, 100.0, (25.0, 75.0)
    )

    file = st.file_uploader('Escolha a base de dados que deseja analisar (.csv)', type='csv')
    if file is not None:
        st.subheader('Estatística descritiva univariada')
        df = pd.read_csv(file)
        aux = pd.DataFrame({"colunas": df.columns, 'tipos': df.dtypes})
        colunas_numericas = list(aux[aux['tipos'] != 'object']['colunas'])
        colunas_object = list(aux[aux['tipos'] == 'object']['colunas'])
        colunas = list(df.columns)
        col = st.selectbox('Selecione a coluna :', colunas_numericas[1:7])
        st.markdown(colunas_numericas)
        if col is not None:
            if col is colunas_numericas[1]:
                st.markdown('A')
            st.markdown('Selecione o que deseja analisar :')
            mean = st.checkbox('Média')
            if mean:
                st.markdown(df[col].mean())
            median = st.checkbox('Mediana')
            if median:
                st.markdown(df[col].median())
            desvio_pad = st.checkbox('Desvio padrão')
            if desvio_pad:
                st.markdown(df[col].std())
            kurtosis = st.checkbox('Kurtosis')
            if kurtosis:
                st.markdown(df[col].kurtosis())
                st.markdown('A coluna está')
            skewness = st.checkbox('Skewness')
            if skewness:
                st.markdown(df[col].skew())
            describe = st.checkbox('Describe')
            if describe:
                st.table(df[colunas_numericas].describe().transpose())
            maximo = st.checkbox('Valor Máximo')
            if maximo:
                st.markdown(df[col].max())
            minimo = st.checkbox('Valor Mínimo')
            if minimo:
                st.markdown(df[col].min())




if __name__ == '__main__':
    main()
