import numbers
import random
from PIL import Image, ImageOps


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, input_image, target_image):
        for transform in self.transforms:
            input_image, target_image = transform(
                input_image,
                target_image,
            )

        return input_image, target_image


class Resize:
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, input_image, target_image):
        target_height, target_width = self.size

        input_image = input_image.resize(
            (target_width, target_height),
            Image.BILINEAR,
        )
        target_image = target_image.resize(
            (target_width, target_height),
            Image.BILINEAR,
        )

        return input_image, target_image


class RandomCrop:
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

        self.padding = padding

    def __call__(self, input_image, target_image):
        if self.padding > 0:
            input_image = ImageOps.expand(
                input_image,
                border=self.padding,
                fill=0,
            )
            target_image = ImageOps.expand(
                target_image,
                border=self.padding,
                fill=0,
            )

        if input_image.size != target_image.size:
            raise ValueError(
                "Input and target images must have the same spatial size."
            )

        width, height = input_image.size
        crop_height, crop_width = self.size

        if width == crop_width and height == crop_height:
            return input_image, target_image

        if width < crop_width or height < crop_height:
            resize_size = (crop_width, crop_height)

            return (
                input_image.resize(resize_size, Image.BILINEAR),
                target_image.resize(resize_size, Image.BILINEAR),
            )

        left = random.randint(0, width - crop_width)
        top = random.randint(0, height - crop_height)

        crop_box = (
            left,
            top,
            left + crop_width,
            top + crop_height,
        )

        return (
            input_image.crop(crop_box),
            target_image.crop(crop_box),
        )


class RandomHorizontalFlip:
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, input_image, target_image):
        if random.random() < self.probability:
            input_image = input_image.transpose(
                Image.FLIP_LEFT_RIGHT
            )
            target_image = target_image.transpose(
                Image.FLIP_LEFT_RIGHT
            )

        return input_image, target_image
