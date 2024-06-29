import zipfile
import os

import numpy as np

import tensorflow as tf
assert tf.__version__.startswith('2')

from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.image_classifier import DataLoader

import matplotlib.pyplot as plt

print("Getting data path..")
# 指定したパスにある dataset.zip ファイルを解凍する
with zipfile.ZipFile('/content/drive/MyDrive/Project/yachogo/dataset.zip', 'r') as zip_ref:
      # encode directory name to cp437, and decode to utf-8
      for info in zip_ref.infolist():
          info.filename = info.filename.encode('cp437').decode('utf-8')
          zip_ref.extract(info, '/content')

# 解凍されたファイルのパスを取得
image_path = '/content/dataset'

print("Loading data..")
data = DataLoader.from_folder(image_path)
train_data, test_data = data.split(0.9)

print("Creating model..")
model = image_classifier.create(train_data)

print("Evaluating model..")
loss, accuracy = model.evaluate(test_data)

print("Exporting model..")
model.export(export_dir='.')