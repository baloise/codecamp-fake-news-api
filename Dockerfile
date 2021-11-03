# DISCLAIMER
# This Dockerfile does NOT follow best practices in creating Dockerfiles or deploying a Flask Python API
# This was done for non-productive demonstration purposes only and will be removed soon

FROM python:3.6-stretch
ENV APP_HOME=/src
ENV APP_USER=appuser
RUN groupadd -r $APP_USER && \
    useradd -r -g $APP_USER -d $APP_HOME -s /sbin/nologin -c "Docker image user" $APP_USER
 
WORKDIR $APP_HOME
COPY requirements.txt .
RUN pip install --trusted-host pypi.python.org -r requirements.txt
COPY ./saved_weights.pt ./
COPY ./main.py ./
EXPOSE 5000:5000
RUN chown -R $APP_USER:$APP_USER $APP_HOME
RUN mkdir /.cache
RUN chown -R $APP_USER:$APP_USER /.cache
RUN chmod -R 777 /.cache
USER $APP_USER
CMD ["python","main.py"]