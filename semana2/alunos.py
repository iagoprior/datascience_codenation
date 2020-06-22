import streamlit as st

def main():
    st.title('Hello World')
    st.header('This is a Header')
    st.subheader('This is a subheadr')
    st.text('This is a text')
    st.image('logo.png')
    st.markdown('Button')
    botao = st.button('Botão')
    if botao:
        st.markdown('Clicado')
    check = st.checkbox('Checkbox')
    st.markdown('Checkbox')
    if check:
        st.markdown('Clicado')
    st.markdown('Radio')
    radio = st.radio('Escolha as opções', ('Opt 1','Opt 2'))
    if radio == 'Opt 1':
        st.markdown('Opt 1')
    if radio == 'Opt 2':
        st.markdown('Opt 2')
    st.markdown('Selectbox')
    select = st.selectbox('Choose opt', ('Opt 1', 'Opt 2'))
    if select == 'Opt 1':
        st.markdown('Opt 1')
    if select == 'Opt 2':
        st.markdown('Opt 2')
    st.markdown('Multi')
    multi = st.multiselect('Choose :', ('Opt 1', 'Opt 2'))
    if multi == 'Opt 1':
        st.markdown('Opt 1')
    if multi == 'Opt 2':
        st.markdown('Opt 2')
    st.markdown('File uploader')
    file = st.file_uploader('Choose your file:', type='csv')
    if file is not None:
        st.markdown('Não está vazio')


if __name__ == '__main__':
    main()
