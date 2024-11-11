# syntax=docker/dockerfile:1
 
FROM python:3.10.12-slim
 
WORKDIR ../AI-101-creating-api
 
# Copy the virtual environment from the parent directory
#COPY ../venv /venv
 
# Set the PATH to use the virtual environment
#ENV PATH="/env/bin:$PATH"
 
COPY requirements.txt .
RUN pip install -r requirements.txt
 
COPY iris_model.pkl .
COPY label_encoder.pkl .
COPY main.py .
COPY project1.ipynb .
 
EXPOSE 5000
 
CMD ["python", "main.py"]