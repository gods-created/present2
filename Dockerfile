FROM python:3.12-slim
COPY . /app
WORKDIR /app
EXPOSE 8001
RUN chmod +x /app/entrypoint.sh
RUN pip install --no-cache-dir -r /app/requirements.txt
RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader punkt_tab
ENTRYPOINT ["/app/entrypoint.sh"]