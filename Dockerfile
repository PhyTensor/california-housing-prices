# Python image
FROM python:3.9-slim
# FROM python:3.9-alpine
# FROM python:3.12-alpine

# set working directory
WORKDIR /app

# install dependencies
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy the rest of the code
COPY ./app .

# expose the port
EXPOSE 8007

# command to run the application with uvicorn
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8007"]

