#CODIGO PARA GUARDAR LA NUEVA PARTICION A MI DRIVE
import os, shutil
from google.colab import drive, files

drive.mount('/content/drive')
os.makedirs('/content/drive/MyDrive/SCRT', exist_ok=True)

shutil.make_archive('/content/drive/MyDrive/SCRT/dataset_por_paciente', 'zip',
                    '/content/dataset_por_paciente')
print("Guardado en Drive: SCRT/dataset_por_paciente.zip")

# files.download('/content/drive/MyDrive/SCRT/dataset_por_paciente.zip')