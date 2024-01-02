# need to add geetools,matplotlib,folium,rasterio,scipy, to env.yml
import ee
import csv
import pandas as pd
from functools import partial
import numpy as np
import time
from geetools import batch
from tqdm import tqdm
from argparse import ArgumentParser
import requests
import matplotlib.pyplot as plt
import folium
import math
import rasterio
import logging
import requests
import shutil
from retry import retry
import multiprocessing
import os
import random

# # initialize GEE using project's service account and JSON key
# service_account = "climateeye@my-project-1571521105802.iam.gserviceaccount.com"
service_account = "app-engine-service-account@capstone-361215.iam.gserviceaccount.com"
json_key = "my.json"
ee.Initialize(
    ee.ServiceAccountCredentials(service_account, json_key),
    opt_url="https://earthengine-highvolume.googleapis.com",
)


def boundingBox(lat, lon, size, res):
    """takes lat, lon of center point, desired size of image,
    and resolution of dataset to return coordinates of
    the four corners of the square centered at (lat, lon) of
    dimensions size

    :param lat: latitude of point of interest
    :type lat: float
    :param lon: longitude of point of interest
    :type lat: float
    :param size: size (in px) of desired image
    :type size: int
    :returns: coordinates (lat, lon) of bounding square corners
    :rtype: float
    """

    earth_radius = 6371000
    angular_distance = math.degrees(0.5 * ((size * res) / earth_radius))
    osLat = angular_distance
    osLon = angular_distance
    xMin = lon - osLon
    xMax = lon + osLon
    yMin = lat - osLat
    yMax = lat + osLat
    return xMin, xMax, yMin, yMax


def time_generator():
    """
    Pick a start data and end date for each location
    """
    date_range = [
        ("2021-01-01", "2021-02-28"),
        ("2021-03-01", "2021-04-30"),
        ("2021-05-01", "2021-06-30"),
        ("2021-07-01", "2021-08-31"),
        ("2021-09-01", "2021-10-30"),
        ("2021-11-01", "2021-12-31"),
    ]
    return random.choice(date_range)


@retry(tries=10, delay=2, backoff=2)
def generateURL(
    coord,
    height,
    width,
    dataset,
    filtered,
    bands,
    crs,
    output_dir,
    dico,
    sharpened=False,
    sar=True,
):
    """generates the URL from Google Earth Engine of the image
    at coordinates coord, from filtered dataset and saves tif file
    to output_dir

    :param coord: longitude and latitude of desired image
    :type coord: tuple or list
    :param height: desired output image height
    :type height: int
    :param width: desired output image width
    :type width: int
    :param dataset: name of dataset (landsat, sentinel, naip)
    :type dataset: str
    :param filtered: filtered Earth Engine dataset (by date)
    :type filtered: ee.ImageCollection()
    :param crs: projection of the image (e.g. EPSG:3857)
    :type crs: str
    :param output_dir: path of output directory
    :type output_dir: str
    :param sharpened: whether we also download pansharpened images too
    :type sharpened: bool
    :param sar: whether we should add sar data
    :type sar: bool
    """

    lon = coord[0]
    lat = coord[1]

    res = dico[dataset]["resolution"]
    # xMin, xMax, yMin, yMax = boundingBox(lat, lon, height, res)

    geometry = ee.Geometry.Rectangle([[coord[6], coord[4]], [coord[5], coord[3]]])
    # geometry = ee.Geometry.Rectangle([[xMin, yMin], [xMax, yMax]])
    filtered = filtered.filterBounds(geometry)
    if dataset == "sentinel":
        cloud_pct = 10
        filtered = filtered.filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
    image = filtered.median().clip(geometry)
    _min = dico[dataset]["min"]
    _max = dico[dataset]["max"]
    l = dico[dataset]["bands"]

    if sar:
        # extend the download bands to include VV and VH from sentinel1
        l = dico[dataset]["bands"] + ["VV", "VH"]
        # getting the sar image collection from the default
        s1 = ee.ImageCollection("COPERNICUS/S1_GRD").filterDate(
            args["start_date"], args["end_date"]
        )
        s1 = s1.filterBounds(geometry)
        s1 = (
            s1.filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .select(["VV", "VH"])
        )
        # if the location has both ascending and descending orbit, choose the orbit that has the most orbits
        asc = s1.filter(ee.Filter.eq("orbitProperties_pass", "ASCENDING"))
        desc = s1.filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
        # select the orbit that has the most images
        if asc.size().getInfo() > desc.size().getInfo():
            image_s1 = asc.median().clip(geometry)
        else:
            image_s1 = desc.median().clip(geometry)
        image = ee.Image(
            [image, image_s1]
        )  # merging the 2 seperate median image together to one

    band_names = image.bandNames()
    bands_list = band_names.getInfo()
    description = f"{dataset}_image_{lat}_{lon}_{l}"

    # only download when all the band exists
    if all(band in bands_list for band in l):
        # if len(bands)!=0:
        try:
            url = image.getDownloadUrl(
                {
                    "image": image.serialize(),
                    "description": description,
                    "region": geometry,
                    "fileNamePrefix": description,
                    "crs": crs,
                    "bands": l,
                    "format": "GEO_TIFF",
                    "dimensions": [height, width],
                }
            )

            # download image given URL
            response = requests.get(url)
            if response.status_code != 200:
                raise response.raise_for_status()
            with open(os.path.join(output_dir, f"{description}.tif"), "wb") as fd:
                fd.write(response.content)
            logging.info(f"Done: {description}")

        except requests.exceptions.RequestException as e:  # Exception as e:
            logging.exception(e)
            logging.info(f"Image at {(lat, lon)} has bands: {bands_list}")
            pass
    else:
        logging.info(f"Image at {(lat, lon)} has bands: {bands_list}")
        pass


def req_func(i, data, url, filtered, output_dir):
    coord = data.iloc[i]
    lon = coord[0]
    lat = coord[1]
    description = f"sentinel_image_{lat}_{lon}"
    # res = dico["sentinel"]["resolution"]
    xMin, xMax, yMin, yMax = boundingBox(lat, lon, 224, 10)

    geometry = ee.Geometry.Rectangle([[xMin, yMin], [xMax, yMax]])

    # filtered = filtered.filterBounds(geometry)

    cloud_pct = 10
    filtered = filtered.filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
    # if (len(filtered.aggregate_array('system:id').getInfo())<=2):

    # print("Number of images in this range : %3d" % len(filtered.aggregate_array('system:id').getInfo()))
    # print(filtered.aggregate_array('system:id').getInfo())
    image = filtered.median().clip(geometry)
    band_names = image.bandNames()
    bands_list = band_names.getInfo()
    url = url[i]
    try:
        s = requests.Session()
        response = s.get(url)
        if response.status_code != 200:
            raise response.raise_for_status()
        with open(os.path.join(f"{output_dir}", f"{description}.tif"), "wb") as fd:
            fd.write(response.content)
        logging.info(f"Done: {description}")
    except Exception as e:
        logging.exception(e)
        logging.info(f"Image at {(lat, lon)} has bands: {bands_list}")
        pass


if __name__ == "__main__":
    args = {
        "dataset": "sentinel",
        "output_dir": "batch/",
        # "output_dir": "/scratch/mh613/tenk_twelve/batch/",  # replace this with ur output dir
        # "filepath": "Data/coordinates.csv",  # replace this with your coordinates files
        "filepath": "clean.csv",
        "start_date": "2022-03-01",
        "end_date": "2022-05-01",
        "sharpened": False,
        "sar": True,
        "width": 224,
        "height": 224,
    }
    print("okay")
    logging.basicConfig(
        filename=f"{args['dataset']}_logger.log",
        filemode="w",
        level="INFO",
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )

    if not os.path.exists(args["output_dir"]):
        os.makedirs(args["output_dir"])
        logging.info(f"Directory {args['output_dir']} created")
    # else:
    #     print("Please delete output directory before retrying")
    # data = pd.read_csv(args["filepath"])
    with open(args["filepath"], "r") as coords_file:
        next(coords_file)
        coords = csv.reader(coords_file, quoting=csv.QUOTE_NONNUMERIC)
        data = list(coords)
    dico = {
        "sentinel": {
            "dataset": ee.ImageCollection("COPERNICUS/S2_SR"),
            "resolution": 10,
            # "bands": ["B2", "B3", "B4"],
            "bands": ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"],
            "min": 0.0,
            "max": 5000.0,
        }
    }
    lat_lon_only = partial(
        generateURL,
        height=args["height"],
        width=args["width"],
        dataset=args["dataset"],
        filtered=dico[args["dataset"]]["dataset"].filterDate(
            args["start_date"], args["end_date"]
        ),
        bands=dico[args["dataset"]]["bands"],
        crs="EPSG:3857",
        output_dir=args["output_dir"],
        dico=dico,
        sharpened=args["sharpened"],
        sar=args["sar"],
    )

    # for i in tqdm(range(0, len(data), 2)):
    #     pool = multiprocessing.Pool()
    #     logging.info(f"Starting rows: {i} to {i+2}")
    #     # export_start_time = time.time()
    #     # print(f"Starting rows: {i} to {i+pn}")
    #     # logging.info(f"Starting rows: {i} to {i+pn}")
    #     start = time.time()
    #     pool.map(lat_lon_only, data[i : i + 2])
    #     time.sleep(1)
    #     # pool.map(on, range(i, i + 10))
    #     end = time.time()
    #     print(end - start)
    #     pool.close()
    #     pool.join()
    export_start_time = time.time()
    for i in tqdm(range(len(data))):
        # Call the download function
        lat_lon_only(data[i])
        # Sleep for 1 second to ensure google quota issue
        time.sleep(1)
        DIR = args.output_dir
        # num_downloaded = len(
        #     [
        #         name
        #         for name in os.listdir(DIR)
        #         if os.path.isfile(os.path.join(DIR, name))
        #     ]
        # )
        logging.info(f"Finished rows: {i}")
        # logging.info(f"Downloaded {num_downloaded} images so far")
        print(f"Finished rows: {i}")
    export_finish_time = time.time()
