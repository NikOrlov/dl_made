# Captcha recognition
### Create environment
Download in project directory `laba-dataset.zip` from yandex disk and then unzip dataset:
~~~
unzip laba-dataset.zip
~~~

Create virtual environment (commands executed in project dir, unix-like systems)

Notion: 
There might be technical issues with torch packages (feel free to go on official site and download available torch package). 

`requirements.txt` include torch packages for CPU-computation (GPU is not supported). You may install GPU-supported torch packages to increase model training. 
~~~
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
~~~
### EDA 
Run in terminal:
```
jupyter-notebook
```
and open `analysis.ipynb`

### Train model:
Run command bellow to train model (in project dir will appear `model.pth`)
~~~
python main.py
~~~
Pretrained weights available in `pretrained_models/*.pth`
