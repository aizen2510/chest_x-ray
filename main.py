import kaggle

kaggle.api.authenticate()
kaggle.api.dataset_download_files('ghost5612/chest-x-ray-images-normal-and-pneumonia', unzip=True, path='.')