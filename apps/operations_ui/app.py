import streamlit as st

st.set_page_config(page_title="AQuant Operations", layout="wide")
st.title("AQuant Operations")
st.error("LIVE 默认关闭")
st.checkbox("人工确认 (界面原型, 不直接改变服务端 LIVE 状态)", value=False)
st.markdown("正式操作必须通过 API 的数据、账户、风险、Kill Switch 和券商联合门控。")
