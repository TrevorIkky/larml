#!/bin/bash
app="larml"
docker build -t ${app} .
docker run -d -p 5001:5000 --name ${app} -v $PWD:/app ${app}
#docker run --rm -v $(pwd):/app composer install
