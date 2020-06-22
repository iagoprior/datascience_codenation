import streamlit as st
import pandas as pd

def main():
    st.title('Dataframe da Iris')
    file = st.file_uploader('Upload your file here:', type='csv')
    if file is not None:
        slider = st.slider('Valores', 1,100)
        df = pd.read_csv(file)
        st.dataframe(df.head(slider))
        st.markdown('Markdown')
        st.table(df.head(slider))
        st.write(df.columns)
        st.table(df.groupby('species')['petal_width'])

if __name__ == '__main__':
    main()