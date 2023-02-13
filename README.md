# mlops_made
### Create environment
Create virtual environment (commands executed in project dir, unix-like systems)

Notion: `requirements.txt` include torch packages for CPU-computation (GPU is not supported). You may install GPU-supported torch packages to increase model training. 
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
and open `notebooks/analysis.ipynb`

### Train model:
Saved parameters available in `model.pth`
~~~
python main.py
~~~