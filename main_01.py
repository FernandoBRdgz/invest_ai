import streamlit as st

st.title("📊 Invest-AI")
st.caption("💰 Inversiones simplificadas con IA")

prompt = st.chat_input("¿En qué te puedo ayudar?")
if prompt:
    st.write("El usuario ha enviado el siguiente prompt: ", prompt)