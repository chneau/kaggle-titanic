.SILENT:
.ONESHELL:
.NOTPARALLEL:
.EXPORT_ALL_VARIABLES:
.PHONY: run kaggle tensor spark

current_dir = $(shell pwd)
target_dir = /script
port = 8080
ip = \*

run: native

spark:
	docker run --rm -it --name=kaggle -p=$(port):$(port) -p=4040:4040 -v=$(current_dir):$(target_dir) -w=$(target_dir) jupyter/pyspark-notebook jupyter notebook --no-browser --notebook-dir=$(target_dir) --allow-root --port=$(port) --ip=$(ip)

kaggle:
	docker run --rm -it --name=kaggle -p=$(port):$(port) -v=$(current_dir):$(target_dir) -w=$(target_dir) kaggle/python jupyter notebook --no-browser --notebook-dir=$(target_dir) --allow-root --port=$(port) --ip=$(ip)

tensor:
	docker run --rm -it --name=tensorflow -p=$(port):$(port) -v=$(current_dir):$(target_dir) -w=$(target_dir) tensorflow/tensorflow jupyter notebook --no-browser --notebook-dir=$(target_dir) --allow-root --port=$(port) --ip=$(ip)

native:
	jupyter notebook --allow-root --port=$(port) --ip=$(ip)
