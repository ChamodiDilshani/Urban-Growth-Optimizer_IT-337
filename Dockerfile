FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir streamlit folium streamlit-folium fpdf2 joblib xgboost scikit-learn shap openpyxl matplotlib pandas numpy
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]