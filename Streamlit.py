import streamlit as st
import pandas as pd
import joblib
import sys
sys.path.append("artifacts")

pipeline_path = "artifacts/preprocessor/preprocessor.pkl"
model_path = "artifacts/model/svm.pkl"
encoder_path = "artifacts/preprocessor/labelencoder.pkl"
with open(pipeline_path, "rb") as file1:
    print(file1.read(100))

try:
    pipeline = joblib.load(pipeline_path)
    print("Pipeline cargado correctamente")
except Exception as e:
    print(f"Error al cargar el pipeline: {e}")

with open(model_path, "rb") as file2:
    print(file2.read(100))

try:
    model = joblib.load(model_path)
    print("Modelo cargado correctamente")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")

with open(encoder_path, "rb") as file3:
    print(file3.read(100))

try:
    encoder = joblib.load(encoder_path)
    print("Encoder cargado correctamente")
except Exception as e:
    print(f"Error al cargar el encoder: {e}")


# Aplicacion 
st.set_page_config(page_title = "Mlweb", layout ="wide")
st.title("Prueba de machine learning con Streamlit")
st.header("Ingreso de datos")

col1, col2, col3 = st.columns(3)
with col1:
    battery_power = st.slider(
        "Poder de la bateria", min_value=500, max_value=2000, value=800
    )
    clock_speed = st.slider(
        "Velocidad del GPU", min_value=0.5, max_value=3.0
    )
    fc = st.slider(
        "Camara frontal", min_value=0, max_value = 19, step=1
    )
    int_memory = st.slider(
        "Memoria interna", min_value=0, max_value=64, value=32,step=1
    )
    px_height = st.slider(
        "Resolucion de la pantalla (alto)", min_value=100, max_value=2000
    )
with col2:
    m_dep = st.slider(
        "Grosor del telefono", min_value=0.1, max_value=1.0
    )
    mobile_wt = st.slider(
        "Peso del telefono", min_value=100, max_value=2000
    )
    n_cores = st.slider(
        "Numero de nucleos", min_value=1, max_value=10
    )
    pc = st.slider(
        "Camara trasera", min_value=0, max_value=19, step=1
    )
    px_width = st.slider(
        "Resolucion de la pantalla (ancho)", min_value=100, max_value=2000
    )
with col3:
    ram = st.slider(
        "Memoria RAM", min_value=256, max_value=4000
    )
    sc_h = st.slider(
        "Altura de la pantalla", min_value=5, max_value=19
    )
    sc_w = st.slider(
        "Ancho de la pantalla", min_value=0, max_value=18
    )
    talk_time = st.slider(
        "Duracion de la bateria bajo uso constante", min_value=2, max_value=20
    )
st.divider()
col4, col5 , col6= st.columns(3)
with col4:
    blue = st.selectbox(
        "Tiene bluetooth", options=["0", "1"]
    )
    three_g = st.selectbox(
        "Tiene 3G", options=["0", "1"]
    )
with col5:
    dual_sim = st.selectbox(
        "Tiene doble SIM", options=["0", "1"]
    )
    touch_screen = st.selectbox(
        "Tiene pantalla tactil", options=["0", "1"]
    )
with col6:
    four_g = st.selectbox(
        "Tiene 4G", options=["0", "1"]
    )
    wifi = st.selectbox(
        "Tiene wifi", options=["0", "1"]
    )

st.divider()

if st.button("Predecir"):
    input_data = pd.DataFrame(
        {
            "battery_power": [battery_power],
            "blue": [blue],
            "clock_speed": [clock_speed],
            "dual_sim": [dual_sim],
            "fc": [fc],
            "four_g": [four_g],
            "int_memory": [int_memory],
            "m_dep": [m_dep],
            "mobile_wt": [mobile_wt],
            "n_cores": [n_cores],
            "pc": [pc],
            "px_height": [px_height],
            "px_width": [px_width],
            "ram": [ram],
            "sc_h": [sc_h],
            "sc_w": [sc_w],
            "talk_time": [talk_time],
            "three_g": [three_g],
            "touch_screen": [touch_screen],
            "wifi": [wifi]
        }
    )

    st.dataframe(input_data)

    pipelined_data =  pipeline.transform(input_data)
    prediction = model.predict(pipelined_data)

    if prediction[0] == 0:
        st.success("El telefono es de gama baja")
    elif prediction[0] == 1:
        st.success("El telefono es de gama media")
    elif prediction[0] == 2:
        st.success("El telefono es de gama alta")
    elif prediction[0] == 3:
        st.success("El telefono es de gama muy alta")
    else:
        st.error("Error en la prediccion")