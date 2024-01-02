# import random
from logging import getLogger
from PIL import ImageFilter
import numpy as np

# import tensorflow as tf
import rasterio
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from .tasks import Task
import torch
from osgeo import gdal

# ... and suppress errors
gdal.PushErrorHandler("CPLQuietErrorHandler")
logger = getLogger()


def rasterio_loader(path: str) -> "np.typing.NDArray[np.int_]":

    with rasterio.open(path) as f:
        array: "np.typing.NDArray[np.int_]" = f.read().astype(np.int32)
        # print(array.shape)
        # tensor = torch.from_numpy(array)
        # print(tensor.size())
        # tensor = tensor.permute((2, 0, 1))
        # NonGeoClassificationDataset expects images returned with channels last (HWC)
        array = array.transpose(1, 2, 0)
    return array / 10000


class TwelveMultiCropDataset(datasets.ImageFolder):
    def __init__(
        self,
        task,
        data_path,
        size_crops,
        nmb_crops,
        min_scale_crops,
        max_scale_crops,
        size_dataset=-1,
        return_index=False,
    ):
        # super(MultiropDataset, self).__init__(data_path)
        super().__init__(
            root=data_path,
            loader=rasterio_loader,
            # IMG_EXTENSIONS if is_valid_file is None else None,
            # transform=transforms,
            # is_valid_file=is_in_split
        )
        # get_color_distortion()
        self.return_index = return_index
        # color_transform = [get_color_distortion()]#PILRandomGaussianBlur()]
        # color_transform=[PILRandomGaussianBlur()]#color_jitter()
        color_transform = [transforms.GaussianBlur(3, sigma=(0.1, 2))]
        task = Task(task)
        mean = task.mean
        std = task.std
        #     mean = [0.17944421, 0.20435592, 0.21557781, 0.25249454, 0.31785015,
        #    0.3416626 , 0.3555974 , 0.36042425, 0.32717814, 0.26588158]
        #     std = [0.10776593, 0.10880394, 0.1191343 , 0.11604542, 0.11204927,
        #    0.11378233, 0.11699808, 0.11209462, 0.10417591, 0.09763059]
        trans = []
        for i in range(len(size_crops)):

            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )
            trans.extend(
                [
                    transforms.Compose(
                        [
                            randomresizedcrop,
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.Compose(color_transform),
                            # transforms.ToTensor(),
                            transforms.Normalize(mean=mean, std=std),
                        ]
                    )
                ]
                * nmb_crops[i]
            )

        self.trans = trans

    def __getitem__(self, index):
        # size_crops = [224, 96]         # [high res, low res]
        # nmb_crops = [2, 6]            # [# high res, # low res]

        # # Experimental options
        # min_scale_crops = [0.14, 0.5]#[0.5, 0.14]
        # max_scale_crops= [1., 0.5]
        path, _ = self.samples[index]
        image = self.loader(path)
        ff = transforms.ToTensor()
        # print(PILRandomGaussianBlur.__call__(image))
        # multi_crops = list(map(lambda x: preprocess(x,size_crops,nmb_crops,min_scale_crops,max_scale_crops), [image]))
        multi_crops = list(map(lambda trans: trans(ff(image)), self.trans))
        # multi_crops = list(map(lambda trans: trans(image), self.trans))
        # if self.return_index:
        #     return index, multi_crops
        return multi_crops


# class PILRandomGaussianBlur(object):
#     """
#     Apply Gaussian Blur to the PIL image. Take the radius and probability of
#     application as the parameter.
#     This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
#     """

#     def __init__(self, p=0.5, kernel_size=3,radius_min=0.1, radius_max=2.):
#         self.prob = p
#         self.radius_min = radius_min
#         self.radius_max = radius_max
#         self.kernel_size=kernel_size
#     def __call__(self, img):
#         yu=transforms.GaussianBlur(
#                 self.kernel_size, sigma=(self.radius_min, self.radius_max)
#             )
#         r=yu(img)
#         op=random_apply(yu,r,self.prob)
#         return torch.tensor(op.numpy())

# class color_jitter(object):
#   def __init__(self, s=1):
#         self.s=s
#   def __call__(self, x):
#         t=random_apply(coloritter,x,0.8)
#         # x = tf.image.random_brightness(x, max_delta=0.8*self.s)
#         # x = tf.image.random_contrast(x, lower=1-0.8*self.s, upper=1+0.8*self.s)
#         # x = tf.clip_by_value(x, 0, 1)
#         return torch.tensor(t.numpy())
# def coloritter(x, s=1):
# 	x = tf.image.random_brightness(x, max_delta=0.8*s)
# 	x = tf.image.random_contrast(x, lower=1-0.8*s, upper=1+0.8*s)
# 	#x = tf.image.random_saturation(x, lower=1-0.8*s, upper=1+0.8*s)
# 	#x = tf.image.random_hue(x, max_delta=0.2*s)
# 	x = tf.clip_by_value(x, 0, 1)
# 	return x
# def random_apply(func, x, p):
# 	return tf.cond(
# 		tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),tf.cast(p, tf.float32)),
#             lambda: func(x),lambda: x)
