#!/bin/bash

app_name=$1

cd ../$app_name/

docker-compose up
