import pandas as pd
import numpy as np
import rasterio
import os
from osgeo import gdal

# ... and suppress errors
# gdal.UseExceptions()
# gdal.PushErrorHandler('CPLQuietErrorHandler')
# takes the file and returns the image array
def get_image_array(file):
    gdal.PushErrorHandler("CPLQuietErrorHandler")
    data = rasterio.open(file)
    image = np.zeros(
        (data.width, data.height, data.count), dtype=data.dtypes[0]
    )  # creates matrix of width X Height X number of bands
    for i in range(data.count - 2):
        image[:, :, i] = data.read(i + 1) / 10000
    for i in range(10, data.count):
        image[:, :, i] = data.read(i + 1)
    # new_image=image[:,:,:]/10000

    return image


gdal.PushErrorHandler("CPLQuietErrorHandler")
twelvechan = "/scratch/mh613/tenk_twelve/batch"  # update path
alle = []
for filena in sorted(os.listdir(twelvechan)):
    alle.append(get_image_array(os.path.join(twelvechan, filena)))  # gets all arrays
newal = [i.reshape(1, 224, 224, 12) for i in alle]  # update 12 to 3 or vice-versa
alp = np.concatenate(newal, axis=0)
print(np.std(alp, axis=(0, 1, 2)))
print(np.mean(alp, axis=(0, 1, 2)))
