# ===============================================================
# app_streamlit_modelos.py
# Demo Streamlit con comparación de modelos de regresión y
# soporte robusto para cargar archivos CSV/XLSX desde la UI.
# (Incluye explicaciones línea por línea como comentarios.)
# ===============================================================

# ----------------------
# 1) IMPORTACIONES
# ----------------------
import streamlit as st
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import partial_dependence, permutation_importance

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor

from sklearn.datasets import load_diabetes, make_regression

import plotly.express as px
import plotly.graph_objects as go

# ----------------------
# 2) CONFIGURACIÓN DE PÁGINA
# ----------------------
st.set_page_config(
    page_title="Comparador de Modelos de Regresión (Streamlit + Plotly)",
    page_icon="📈",
    layout="wide"
)

# ----------------------
# 3) ENCABEZADO
# ----------------------
st.title("📈 Comparador de Modelos de Regresión")
st.markdown(
    """
Este demo permite **cargar un dataset (CSV/XLSX)**, **entrenar múltiples modelos** y 
**visualizar** correlaciones, **pred vs real**, **residuales**, **importancia de variables** y **dependencia parcial**.
"""
)

# ----------------------
# 4) HELPERS: CARGA DE ARCHIVOS
# ----------------------
@st.cache_data(show_spinner=False)
def read_csv_bytes(file_bytes: bytes, delimiter: str) -> pd.DataFrame:
    # Lee CSV desde bytes con un delimitador elegido por el usuario
    from io import BytesIO, StringIO
    try:
        # Intento directo como texto
        return pd.read_csv(StringIO(file_bytes.decode('utf-8')), sep=delimiter)
    except Exception:
        # Fallback a binario
        return pd.read_csv(BytesIO(file_bytes), sep=delimiter)

@st.cache_data(show_spinner=False)
def read_excel_bytes(file_bytes: bytes, sheet_name: Optional[str]) -> pd.DataFrame:
    # Lee Excel desde bytes; si no hay sheet_name se toma la primera hoja
    import io
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    if sheet_name is None or sheet_name not in xls.sheet_names:
        sheet_name = xls.sheet_names[0]
    return pd.read_excel(xls, sheet_name=sheet_name)

def build_data_dict_from_df(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    # Asegura que solo trabajamos con columnas numéricas para sklearn
    if target_col not in df.columns:
        raise ValueError("La columna objetivo seleccionada no existe en el DataFrame.")
    y = df[target_col]
    X = df.drop(columns=[target_col])
    # Filtra a numéricas (si hay categóricas, sugiere preprocesarlas fuera o convertir dummy aparte)
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_cols) < X.shape[1]:
        st.warning("Se detectaron columnas no numéricas. Solo se usarán columnas numéricas para los modelos.")
    X = X[numeric_cols]
    # Elimina filas con NA en X o y (decisión simple para el demo)
    valid_idx = X.notna().all(axis=1) & y.notna()
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]
    df_clean = X.copy()
    df_clean[target_col] = y
    return {"X": X, "y": y, "df": df_clean, "feature_names": list(X.columns), "target_name": target_col}

# ----------------------
# 5) SIDEBAR: CARGA DE ARCHIVO O DATASETS DE EJEMPLO
# ----------------------
st.sidebar.header("📂 Datos")

# Bloque de subida de archivo
file = st.sidebar.file_uploader(
    "Sube tu dataset (CSV o XLSX)", type=["csv", "xlsx"]
)

data_dict: Optional[Dict[str, Any]] = None
uploaded_target: Optional[str] = None

if file is not None:
    # Si es CSV, pedimos delimitador
    if file.name.lower().endswith(".csv"):
        delimiter = st.sidebar.selectbox("Delimitador (CSV)", [",", ";", "\t", "|"], index=0)
        df_up = read_csv_bytes(file.getvalue(), delimiter)
    else:
        # XLSX: elegimos hoja
        import io
        xls = pd.ExcelFile(io.BytesIO(file.getvalue()))
        sheet = st.sidebar.selectbox("Hoja (XLSX)", xls.sheet_names, index=0)
        df_up = read_excel_bytes(file.getvalue(), sheet)

    st.sidebar.success(f"Archivo cargado: {file.name} ({len(df_up)} filas, {len(df_up.columns)} columnas)")

    # Selección de columna objetivo
    uploaded_target = st.sidebar.selectbox("Selecciona la columna objetivo (y)", df_up.columns)

    # Construimos data_dict desde el archivo subido
    try:
        data_dict = build_data_dict_from_df(df_up, uploaded_target)
    except Exception as e:
        st.sidebar.error(f"Error preparando datos: {e}")
        st.stop()

else:
    # Si no subieron archivo, ofrecer datasets de ejemplo
    dataset_choice = st.sidebar.selectbox("O elige un dataset de ejemplo", 
                                          ["Diabetes (sklearn)", "Sintético (make_regression)", "Sintético de viviendas (CSV de ejemplo)"])

    def load_dataset(choice: str) -> Dict[str, Any]:
        if choice == "Diabetes (sklearn)":
            data = load_diabetes(as_frame=True)
            df = data.frame.copy()
            X = df.drop(columns=[data.target_names[0]])
            y = df[data.target_names[0]]
            return {"X": X, "y": y, "df": df, "feature_names": list(X.columns), "target_name": data.target_names[0]}
        elif choice == "Sintético (make_regression)":
            n_samples = st.sidebar.slider("n_samples", 200, 3000, 900, 100)
            n_features = st.sidebar.slider("n_features", 4, 30, 8)
            noise = st.sidebar.slider("noise", 0.0, 30.0, 8.0)
            X_arr, y_arr = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=42)
            feature_names = [f"feat_{i}" for i in range(n_features)]
            X = pd.DataFrame(X_arr, columns=feature_names)
            y = pd.Series(y_arr, name="target")
            df = X.copy()
            df["target"] = y
            return {"X": X, "y": y, "df": df, "feature_names": feature_names, "target_name": "target"}
        else:
            # Carga el CSV de viviendas previamente generado si está disponible en el entorno
            try:
                df = pd.read_csv("/mnt/data/dataset_regresion_viviendas_1500.csv")
            except Exception:
                st.warning("No se encontró el CSV de ejemplo en el entorno. Elige otro dataset o sube tu archivo.")
                st.stop()
            X = df.drop(columns=["price_usd"])
            y = df["price_usd"]
            return {"X": X, "y": y, "df": df, "feature_names": list(X.columns), "target_name": "price_usd"}

    data_dict = load_dataset(dataset_choice)

# Variables comunes a partir de data_dict
X, y = data_dict["X"], data_dict["y"]
feature_names = data_dict["feature_names"]
target_name = data_dict["target_name"]

# ----------------------
# 6) VISTA PREVIA DEL DATASET
# ----------------------
with st.expander("👀 Vista previa del dataset"):
    st.write(data_dict["df"].head())

# ----------------------
# 7) CONFIGURACIÓN DE SPLIT Y CV
# ----------------------
st.sidebar.header("⚙️ Entrenamiento")
test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2)
random_state = st.sidebar.number_input("Random state", 0, 10_000, 42)
do_cv = st.sidebar.checkbox("Calcular CV (K-Fold)", value=False)
k_folds = st.sidebar.slider("K folds", 3, 10, 5)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# ----------------------
# 8) MODELADO
# ----------------------
st.sidebar.subheader("🧠 Modelos")
poly_degree = st.sidebar.slider("Polynomial degree", 2, 6, 3)
alpha_ridge = st.sidebar.slider("Ridge alpha", 0.0001, 10.0, 1.0, log=True)
alpha_lasso = st.sidebar.slider("Lasso alpha", 0.0001, 10.0, 0.1, log=True)
alpha_en = st.sidebar.slider("ElasticNet alpha", 0.0001, 10.0, 0.5, log=True)
l1_ratio_en = st.sidebar.slider("ElasticNet l1_ratio", 0.0, 1.0, 0.5)
svr_C = st.sidebar.slider("SVR C", 0.1, 100.0, 10.0, log=True)
svr_epsilon = st.sidebar.slider("SVR epsilon", 0.01, 1.0, 0.1, log=True)
svr_gamma = st.sidebar.selectbox("SVR gamma", ["scale", "auto"])
knn_k = st.sidebar.slider("KNN k", 1, 30, 5)
tree_max_depth = st.sidebar.slider("Árbol max_depth", 1, 30, 6)
rf_n_estimators = st.sidebar.slider("RandomForest n_estimators", 50, 500, 200, 50)
rf_max_depth = st.sidebar.slider("RandomForest max_depth", 2, 30, 10)
gb_n_estimators = st.sidebar.slider("GB n_estimators", 50, 500, 200, 50)
gb_learning_rate = st.sidebar.slider("GB learning_rate", 0.01, 0.5, 0.1)
gb_max_depth = st.sidebar.slider("GB max_depth", 2, 10, 3)
et_n_estimators = st.sidebar.slider("ExtraTrees n_estimators", 50, 500, 200, 50)
et_max_depth = st.sidebar.slider("ExtraTrees max_depth", 2, 30, 12)

def build_models() -> Dict[str, Pipeline]:
    standard = ("scale", StandardScaler(), feature_names)
    pre = ColumnTransformer(transformers=[standard], remainder="passthrough")

    models: Dict[str, Pipeline] = {}
    models["Linear"] = Pipeline([("prep", pre), ("lin", LinearRegression())])
    models["Polynomial"] = Pipeline([("prep", pre), ("poly", PolynomialFeatures(degree=poly_degree, include_bias=False)),
                                     ("scale2", StandardScaler(with_mean=False)), ("lin", LinearRegression())])
    models["Ridge"] = Pipeline([("prep", pre), ("ridge", Ridge(alpha=alpha_ridge, random_state=random_state))])
    models["Lasso"] = Pipeline([("prep", pre), ("lasso", Lasso(alpha=alpha_lasso, random_state=random_state, max_iter=20_000))])
    models["ElasticNet"] = Pipeline([("prep", pre), ("enet", ElasticNet(alpha=alpha_en, l1_ratio=l1_ratio_en, random_state=random_state, max_iter=20_000))])
    models["SVR (RBF)"] = Pipeline([("prep", pre), ("svr", SVR(C=svr_C, epsilon=svr_epsilon, gamma=svr_gamma))])
    models["KNN"] = Pipeline([("prep", pre), ("knn", KNeighborsRegressor(n_neighbors=knn_k))])
    models["DecisionTree"] = Pipeline([("tree", DecisionTreeRegressor(max_depth=tree_max_depth, random_state=random_state))])
    models["RandomForest"] = Pipeline([("rf", RandomForestRegressor(n_estimators=rf_n_estimators, max_depth=rf_max_depth, random_state=random_state, n_jobs=-1))])
    models["GradientBoosting"] = Pipeline([("gb", GradientBoostingRegressor(n_estimators=gb_n_estimators, learning_rate=gb_learning_rate, max_depth=gb_max_depth, random_state=random_state))])
    models["ExtraTrees"] = Pipeline([("et", ExtraTreesRegressor(n_estimators=et_n_estimators, max_depth=et_max_depth, random_state=random_state, n_jobs=-1))])
    return models

all_models = build_models()
model_names = list(all_models.keys())
selected_models = st.sidebar.multiselect("Selecciona modelos", model_names, default=["Linear", "RandomForest", "GradientBoosting"])

# ----------------------
# 9) ENTRENAMIENTO + MÉTRICAS
# ----------------------
results: List[Dict[str, Any]] = []
fitted_models: Dict[str, Pipeline] = {}
preds_store: Dict[str, np.ndarray] = {}

for name in selected_models:
    model = all_models[name]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    if do_cv:
        cv = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="r2", n_jobs=-1)
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
    else:
        cv_mean, cv_std = np.nan, np.nan

    results.append({"Modelo": name, "R2": r2, "MAE": mae, "RMSE": rmse, "CV_R2_mean": cv_mean, "CV_R2_std": cv_std})
    fitted_models[name] = model
    preds_store[name] = y_pred

if results:
    metrics_df = pd.DataFrame(results).sort_values("R2", ascending=False)
    st.subheader("🏁 Resultados (Test)")
    st.dataframe(metrics_df, use_container_width=True)

# ----------------------
# 10) HEATMAP DE CORRELACIÓN
# ----------------------
with st.expander("📊 Correlación (heatmap)"):
    df_corr = data_dict["df"].corr(numeric_only=True)
    fig_corr = px.imshow(df_corr, text_auto=".2f", aspect="auto", color_continuous_scale="Viridis", title="Matriz de correlación")
    st.plotly_chart(fig_corr, use_container_width=True)

# ----------------------
# 11) ANÁLISIS DETALLADO POR MODELO
# ----------------------
st.subheader("🔎 Análisis detallado por modelo")
if not selected_models:
    st.info("Elige al menos un modelo en la barra lateral.")
else:
    focus_model_name = st.selectbox("Modelo para análisis profundo", selected_models)
    model = fitted_models[focus_model_name]
    y_pred = preds_store[focus_model_name]

    # 11.1) Predicho vs Real
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers", name="Predicho vs Real", opacity=0.7))
    min_axis = float(min(y_test.min(), y_pred.min()))
    max_axis = float(max(y_test.max(), y_pred.max()))
    fig_scatter.add_trace(go.Scatter(x=[min_axis, max_axis], y=[min_axis, max_axis], mode="lines", name="Identidad (y = x)"))
    fig_scatter.update_layout(title=f"Predicho vs Real — {focus_model_name}", xaxis_title="Real", yaxis_title="Predicho", template="plotly_white")
    st.plotly_chart(fig_scatter, use_container_width=True)

    # 11.2) Residuales
    residuals = y_test - y_pred
    fig_res_hist = px.histogram(residuals, nbins=30, title=f"Distribución de residuales — {focus_model_name}")
    fig_res_hist.update_layout(template="plotly_white")
    st.plotly_chart(fig_res_hist, use_container_width=True)

    fig_res_scatter = px.scatter(x=y_pred, y=residuals, labels={"x": "Predicho", "y": "Residual"}, title=f"Residual vs Predicho — {focus_model_name}")
    fig_res_scatter.add_hline(y=0, line_dash="dash")
    fig_res_scatter.update_layout(template="plotly_white")
    st.plotly_chart(fig_res_scatter, use_container_width=True)

    # 11.3) Importancia de variables
    st.markdown("#### 📌 Importancia de variables")
    def get_feature_importance(estimator, Xv, yv) -> pd.DataFrame:
        final_step = estimator.steps[-1][1] if isinstance(estimator, Pipeline) else estimator
        if hasattr(final_step, "feature_importances_"):
            importances = final_step.feature_importances_
            names = feature_names
            return pd.DataFrame({"feature": names, "importance": importances}).sort_values("importance", ascending=False)
        r = permutation_importance(estimator, Xv, yv, n_repeats=10, random_state=random_state, n_jobs=-1)
        names = feature_names
        return pd.DataFrame({"feature": names, "importance": r.importances_mean}).sort_values("importance", ascending=False)

    try:
        fi_df = get_feature_importance(model, X_test, y_test)
        fig_imp = px.bar(fi_df.head(20), x="importance", y="feature", orientation="h", title=f"Top Importancias — {focus_model_name}")
        fig_imp.update_layout(template="plotly_white")
        st.plotly_chart(fig_imp, use_container_width=True)
    except Exception as e:
        st.warning(f"No fue posible calcular importancia de variables: {e}")

    # 11.4) Dependencia Parcial (1D)
    st.markdown("#### 🧭 Dependencia parcial (1D)")
    pd_feature = st.selectbox("Elige una característica", feature_names)
    try:
        pd_result = partial_dependence(model, X=X_test, features=[feature_names.index(pd_feature)], kind="average")
        xs = pd_result["values"][0]
        ys = pd_result["average"][0]
        fig_pd = px.line(x=xs, y=ys, labels={"x": pd_feature, "y": "Predicción media"}, title=f"Dependencia parcial — {focus_model_name} sobre {pd_feature}")
        fig_pd.update_layout(template="plotly_white")
        st.plotly_chart(fig_pd, use_container_width=True)
    except Exception as e:
        st.info(f"No se pudo calcular dependencia parcial para este modelo/feature: {e}")

# ----------------------
# 12) PIE DE PÁGINA + DESCARGA CSV DE EJEMPLO
# ----------------------
st.caption("Creado con ❤️ usando Streamlit, scikit-learn y Plotly.")
try:
    # Si existe el CSV de ejemplo, ofrecer descarga
    with open("/mnt/data/dataset_regresion_viviendas_1500.csv", "rb") as f:
        st.download_button("⬇️ Descargar CSV de ejemplo (viviendas, 1.500 filas)", f, file_name="dataset_regresion_viviendas_1500.csv", mime="text/csv")
except Exception:
    pass
