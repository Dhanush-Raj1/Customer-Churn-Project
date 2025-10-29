FROM python:3.9-slim

COPY . /app 

WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE 8000

# for development
# CMD ["python", "app.py"]

# for production
#CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--threads", "2", "--timeout", "60", "app:app"]

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--threads", "2", "--timeout", "60", "--log-level", "debug", "--access-logfile", "-", "--error-logfile", "-", "app:app"]