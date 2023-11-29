FROM python:3.9

ADD requirements.txt /requirements.txt

# Install requirements including gdown
RUN pip3 install -r /requirements.txt \
    && rm -Rf /var/cache/apt \
    && rm -Rf /requirements.txt \
    && rm -Rf /root/.cache/pip

# Download the model file from Google Drive
RUN mkdir /app \
	&& gdown --id 1B1VqoItBMwRpU0Avuus7HzrNvfW-evur -O /app/sk_model7.pkl

ENV PYTHONPATH=/app

# Add all files from the current directory to /app/ in the container
ADD * /app/

ENTRYPOINT [ "python3", "/app/sklearn_baseline.py", "-i", "$inputDataset/inputs.jsonl", "-o", "$outputDir", "-m", "/app/sk_model.pkl" ]
