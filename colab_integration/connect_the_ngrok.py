#@title Connect the NGROK

import os
import shutil
from IPython import get_ipython


def connect_the_ngrok():
    if not os.path.exists('/content'):
        print('Not in COLAB')
        return

    ngrok_disk_path = '/content/drive/My Drive/Colab Notebooks/ngrok'
    ngrok_local_path = '/content/ngrok'

    if not os.path.exists(ngrok_local_path):
        print('ngrok is not found, copying to the ', ngrok_local_path, end='...')
        shutil.copy(ngrok_disk_path, '/content')
        print('copied!')
    else:
        print('ngrok already exists in the ', ngrok_local_path)

    os.chdir('/content')
    get_ipython().system_raw('! chmod 755 ngrok &')
    get_ipython().system_raw('! ./ngrok authtoken ... &')
    get_ipython().system_raw('./ngrok http 5000 & ')


connect_the_ngrok()

os.makedirs('mlflow_dir/mlflow')
os.chdir('mlflow_dir')
get_ipython().system_raw('mlflow ui &')
