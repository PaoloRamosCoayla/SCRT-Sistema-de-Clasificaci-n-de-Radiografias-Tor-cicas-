#CODIGO PARA COPIAR EL DATASET DE MI DRIVE AL ENTORNO LOCAL
from google.colab import drive
drive.mount('/content/drive')
#!unzip -q /content/drive/MyDrive/SCRT/dataset_por_paciente.zip -d /content/dataset_por_paciente
#el unzip se ejecuta en colab