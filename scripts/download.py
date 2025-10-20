import os
import kagglehub

#recipen = kagglehub.dataset_download("paultimothymooney/recipenlg") # save to cache
#os.rename(recipen, "../data/raw/recipenlg") # move to local
# Is's too large

recipe = kagglehub.dataset_download("pes12017000148/food-ingredients-and-recipe-dataset-with-images") # save to cache
os.rename(recipe, "../data/raw/pes12017000148") # move to local