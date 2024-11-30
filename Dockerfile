FROM python:3.10.12
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "--workers=2", "--timeout=600", "--bind", "0.0.0.0:5000", "app:app"]
