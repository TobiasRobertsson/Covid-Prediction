FROM python:3.9-slim

COPY ./requirements.txt ./requirements.txt

# pip-installs
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r "requirements.txt"

COPY ./ ./

EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]
CMD ["ml.py"]