import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
import plotly.express as px
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import streamlit as st
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
import streamlit as st
import plotly.express as px
from sklearn.ensemble import IsolationForest

def show_cleaned_series(df, label: str):
    st.subheader(f"🧹 Sygnał oczyszczony z anomalii: {label}")

    if label not in df.columns:
        st.warning(f"Kolumna '{label}' nie istnieje.")
        return
    if "_time" not in df.columns:
        st.warning("Brakuje kolumny '_time' w danych.")
        return

    df_clean = df[["_time", label]].dropna()
    if df_clean.empty:
        st.warning("Brak danych do oczyszczenia.")
        return

    # --- IsolationForest ---
    clf = IsolationForest(contamination=0.4, random_state=42)
    mask_iforest = clf.fit_predict(df_clean[[label]]) == 1

    # --- Reguły dodatkowe ---
    series = df_clean[label]
    std_dev = series.std()
    mean = series.mean()
    upper_limit = mean + 1 * std_dev  # lub 2 std jeśli chcesz ostrzej

    mask_rules = (series >= 0) & (series <= upper_limit)

    # --- Połączone maski ---
    final_mask = mask_iforest & mask_rules
    cleaned = df_clean[final_mask]

    st.info(f"📉 Usunięto {len(df_clean) - len(cleaned)} punktów odstających")

    fig = px.line(cleaned, x="_time", y=label, title=f"Oczyszczony sygnał: {label}")
    fig.update_layout(yaxis_title=label)
    st.plotly_chart(fig, use_container_width=True)


def show_anomalies(df, label: str):
    if label not in df.columns:
        st.warning(f"Kolumna '{label}' nie istnieje w danych.")
        return

    st.subheader(f"🚨 Detekcja anomalii: {label}")

    # Przygotuj dane
    data = df[["_time", label]].dropna()
    if data.empty:
        st.warning("Brak danych do detekcji anomalii.")
        return

    # Model izolacyjny
    model = IsolationForest(contamination=0.05, random_state=42)
    data["anomaly"] = model.fit_predict(data[[label]])

    # Wykres Plotly
    fig = go.Figure()

    # Wartości normalne
    normal = data[data["anomaly"] == 1]
    fig.add_trace(go.Scatter(
        x=normal["_time"],
        y=normal[label],
        mode="lines+markers",
        name="Normalne",
        line=dict(color="blue")
    ))

    # Anomalie
    anomalies = data[data["anomaly"] == -1]
    fig.add_trace(go.Scatter(
        x=anomalies["_time"],
        y=anomalies[label],
        mode="markers",
        name="Anomalie",
        marker=dict(color="red", size=10, symbol="x")
    ))

    fig.update_layout(
        title=f"Anomalie w {label}",
        xaxis_title="Czas",
        yaxis_title=label,
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

def show_correlation(df):
    st.subheader("📊 Macierz korelacji")
    
    # Użyj tylko kolumn numerycznych
    numeric_df = df.select_dtypes(include=['number'])
    
    # Sprawdzenie, czy są wystarczające dane
    if numeric_df.shape[1] < 2:
        st.warning("Za mało danych numerycznych do obliczenia korelacji.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)


