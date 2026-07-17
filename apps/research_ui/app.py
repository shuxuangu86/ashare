import streamlit as st

st.set_page_config(page_title="AQuant Research", layout="wide")
st.title("AQuant Research")
st.warning("研究界面不会直接提交券商订单。")
st.metric("运行模式", "BACKTEST")
st.markdown("因子、模型和回测结果必须绑定数据发布版本后才可展示为可复现实验。")
