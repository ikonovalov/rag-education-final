import os
import kagglehub

#recipen = kagglehub.dataset_download("paultimothymooney/recipenlg") # save to cache
#os.rename(recipen, "../data/raw/recipenlg") # move to local
# Too large

recipen = kagglehub.dataset_download("pes12017000148/food-ingredients-and-recipe-dataset-with-images") # save to cache
os.rename(recipen, "../data/raw/pes12017000148") # move to local