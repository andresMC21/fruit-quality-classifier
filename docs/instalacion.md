# Instalación
1. Python 3.14 (versión con la que se verificó todo) y un entorno virtual: `py -3.14 -m venv .venv` y luego `.venv\Scripts\activate` (Windows).
2. Con GPU NVIDIA, instalar primero PyTorch con CUDA:
   `pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128`.
   Comprobar con `python -c "import torch; print(torch.cuda.is_available())"`. Sin GPU, el paso 3 instala la versión para CPU.
3. `pip install -r requirements.txt`. Las versiones exactas probadas están en `requirements-lock.txt`.
4. Descargar el dataset y extraer las manzanas. La ficha de Kaggle pide cuenta; el mismo contenido (FruitNet, CC BY 4.0) se baja sin cuenta desde Mendeley:
   ```powershell
   curl.exe -L -C - -o data\raw\kaggle\FruitNetDataset.zip "https://data.mendeley.com/public-files/datasets/b6fftwbr2v/files/532e56bf-d855-4afd-86fd-2c8c425eb32c/file_downloaded"
   python -m src.data.fruitnet_inventory data\raw\kaggle\FruitNetDataset.zip --extract-apple
   ```
   Queda buena/mala en `data/raw/kaggle/` y la carpeta de calidad mixta en `data/raw/mixed_quality/Apple/`.
5. Descargar las imágenes propias y las de otros grupos desde la carpeta compartida (enlace en `data/README.md`).
6. Ejecutar los comandos del README desde la raíz del repositorio. Pruebas: `python -m pytest -q`.
