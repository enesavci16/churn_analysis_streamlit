FROM python:3.10.7
COPY pythonProject_churn_analysis/app
COPY /app/requirements.txt /app/
RUN pip install -r /app/requirements.txt
COPY ./pythonProject_churn_analysis/main /model/
COPY ./app /app
Footer
