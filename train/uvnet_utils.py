# coding=utf-8

"""Misc util objects and functions for the uv-net."""

import cntk
import copy
import cv2
import jinja2
import math
import numpy as np
import os
import pickle
import pprint
import pygments
import random
import re
import string
import subprocess
import sys

from enum import Enum
from PIL import Image, ImageDraw, ImageFont
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer
from typing import Any, List, Tuple, Union


class DataTypes(Enum):
    """Different data types used internally in uv-net."""
    Input = 1
    Uv = 2
    Mask = 3


class ModelParameters:
    """Class for holding model parameters."""

    def __init__(self):
        self.levels = 6
        self.filter_size = (5, 5)
        self.initial_features = 32
        self.feature_multiplier = 1.75
        self.up_factor = 1.25


class AugmentParameters:
    """Class for holding data augmentation parameters."""

    def __init__(self):
        self.enabled = True
        self.occlusion = True
        self.shuffle = True
        self.rotate = False
        self.exposure = True
        self.gamma = True
        self.noise = True
        self.clip = True
        self.quantize = True

        self.occlusion_prob = 0.5
        self.shuffle_prob = 0.5
        self.rotate_prob = 0.5
        self.exposure_min = 0.5
        self.exposure_max = 2.0
        self.gamma_min = 0.5
        self.gamma_max = 1.5
        self.noise_mean = 0.0
        self.noise_std = 0.333
        self.noise_scale_min = 0.0
        self.noise_scale_max = 0.15
        self.quantization_levels = 255.0
        self.mask_erosion_amount = 3


class LearnerParameters:
    """Class for holding learner specific parameters."""

    def __init__(self):
        self.learning_rate = 0.0001  # 0.0001
        self.momentum = 0.9  # 0.9
        self.l1_regularization_weight = 0.0
        self.l2_regularization_weight = 0.0
        self.gaussian_noise_injection_std_dev = 0.0  # 0.0001 or smaller
        self.gradient_clipping_threshold_per_sample = np.inf
        self.gradient_clipping_with_truncation = True
        self.use_mean_gradient = False


class Parameters:
    """Class for holding all training parameters."""

    def __init__(self):
        self.run_description = "base"

        self.output_directory = ""
        self.textures_directory = ""
        self.train_image_directories = []
        self.test_image_directories = []
        self.real_image_aligned_directories = []
        self.real_image_directories = []
        self.train_image_filter = ".exr"
        self.test_image_filter = ".exr"
        self.input_size = (128, 128)

        self.train_epoch_count = 10000
        self.train_epoch_size = 10000
        self.train_minibatch_size = 40
        self.test_epoch_size = 1000
        self.test_minibatch_size = 40

        self.max_training_time = 60 * 60 * 71  # in seconds

        self.model_load_from_file = False
        self.model_load_file_name = ""
        self.model_save_intermediate = True
        self.model_save_interval = 100  # in epochs
        self.model_save_final = True
        self.use_random_eval_augment = True

        self.model = ModelParameters()
        self.train_augment = AugmentParameters()
        self.test_augment = AugmentParameters()
        self.eval_augment = AugmentParameters()
        self.learner = LearnerParameters()

        self.eval_augment.enabled = True
        self.eval_augment.occlusion = True
        self.eval_augment.shuffle = False
        self.eval_augment.rotate = False
        self.eval_augment.exposure = False
        self.eval_augment.gamma = False
        self.eval_augment.noise = False
        self.eval_augment.clip = True
        self.eval_augment.quantize = True

    def to_string(self) -> str:
        """Convert class to a string representation."""
        c = copy.deepcopy(self)
        v = vars(c)
        v["learner"] = vars(c.learner)
        v["model"] = vars(c.model)
        v["test_augment"] = vars(c.test_augment)
        v["train_augment"] = vars(c.train_augment)
        v["eval_augment"] = vars(c.eval_augment)
        return pprint.pformat(v)

    def save_to_file(self, file_path: str) -> None:
        """Serialize parameters to a pickle file."""
        pickle.dump(self, open(file_path, "wb"))


class Logger:
    """Class used for mirroring stdout to a file."""

    def __init__(self, name: str, mode: str):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, text: str):
        """Write to both a file and the stdout."""
        self.file.write(text)
        self.file.flush()
        self.stdout.write(text)

    def flush(self):
        """Flush data when closing."""
        self.file.flush()


def load_parameters_from_file(file_path: str) -> Parameters:
    """Load serialized parameters from a pickle file."""
    params = pickle.load(open(file_path, "rb"))
    return params


def clamp(x: Union[int, float], min_: Union[int, float], max_: Union[int, float]) -> Union[int, float]:
    """Clamp value between min and max."""
    return max(min_, min(x, max_))


def cntk_l1_loss(a: Union[cntk.Function, cntk.Variable], b: Union[cntk.Function, cntk.Variable]) -> cntk.Function:
    """Calculate L1 loss between tensors."""
    return cntk.ops.reduce_sum(cntk.ops.abs(a - b))


def cntk_l2_loss(a: Union[cntk.Function, cntk.Variable], b: Union[cntk.Function, cntk.Variable]) -> cntk.Function:
    """Calculate L2 loss between tensors."""
    return cntk.losses.squared_error(a, b)


def cntk_l1_fine(a: Union[cntk.Function, cntk.Variable]) -> cntk.Function:
    """Calculate L1 fine of a tensor."""
    return cntk.ops.reduce_sum(cntk.ops.abs(a))


def cntk_l2_fine(a: Union[cntk.Function, cntk.Variable]) -> cntk.Function:
    """Calculate L2 fine of a tensor."""
    return cntk.ops.reduce_sum(a * a)


def intspace(value: Union[int, str]) -> str:
    """Converts an integer to a string containing spaces every three digits (from https://github.com/jmoiron/humanize)."""
    orig = str(value)
    new = re.sub("^(-?\d+)(\d{3})", '\g<1> \g<2>', orig)
    if orig == new:
        return new
    else:
        return intspace(new)


def tuple_product(t: Tuple) -> int:
    """Calculate the product of the tuple elements."""
    result = 1

    for v in t:
        result = result * v

    return result


def calculate_parameter_count(model: cntk.Function) -> int:
    """Calculate the total amount of trainable parameters in a model."""
    param_count = 0

    for p in model.parameters:
        param_count += tuple_product(p.shape)

    return param_count


def get_sched_value(value_schedule: Union[Any, List[Tuple[Any, int]]], epoch_number: int) -> Any:
    """Get a value that can be scheduled to change based on the epoch number.
    value_schedule is either a scalar value or a list of tuples [(epochs, value1), (epochs, value2), ...]"""
    if not isinstance(value_schedule, list):
        return value_schedule
    else:
        epoch_sum = 0

        for v in value_schedule:
            epoch_sum += v[0]

            if epoch_number <= epoch_sum:
                return v[1]

        return value_schedule[-1][1]


def get_random_string(length: int) -> str:
    """Generate random string of given length consisting of ASCII letters and numbers."""
    pool = string.ascii_letters + string.digits
    return "".join(random.choice(pool) for _ in range(length))


def swap_to_front(data: np.ndarray) -> np.ndarray:
    """Swap color channels from back to front."""
    return np.swapaxes(np.swapaxes(data, 1, 2), 0, 1)


def swap_to_back(data: np.ndarray) -> np.ndarray:
    """Swap color channels from front to back."""
    return np.swapaxes(np.swapaxes(data, 0, 1), 1, 2)


def get_git_commit_tag_name() -> Union[str, None]:
    """Get the current git commit tag name."""
    result = subprocess.run(["git", "describe", "--tags", "--abbrev=0", "--always", "--exact-match"], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode == 0:
        return result.stdout.decode(sys.stdout.encoding).strip()
    else:
        return None


def get_git_commit_short_hash() -> Union[str, None]:
    """Get the current git commit short hash."""
    result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode == 0:
        return result.stdout.decode(sys.stdout.encoding).strip()
    else:
        return None


def read_file_paths(dir_path: str, end_filter: str = None) -> List[str]:
    """Read file paths in a directory ending with a filter string. Returns absolute paths."""
    print("Reading file paths from {0}".format(dir_path))
    paths = os.listdir(dir_path)
    paths = [os.path.join(dir_path, p) for p in paths]

    if end_filter is not None:
        paths = sorted([p for p in paths if os.path.isfile(p) and p.endswith(end_filter)])
    else:
        paths = sorted([p for p in paths if os.path.isfile(p)])

    return paths


def read_file_paths_as_triplets(dir_path: str, end_filter: str = None) -> List[Tuple[str, str, str]]:
    """Read file paths in a directory ending with a filter string and zip them into tuples containing three consecutive (sorted alphabetically) items. Returns absolute paths."""
    print("Reading file paths from {0}".format(dir_path))
    paths = os.listdir(dir_path)
    paths = [os.path.join(dir_path, p) for p in paths]

    if end_filter is not None:
        paths = sorted([p for p in paths if os.path.isfile(p) and p.endswith(end_filter)])
    else:
        paths = sorted([p for p in paths if os.path.isfile(p)])

    assert len(paths) % 3 == 0
    return list(zip(paths[::3], paths[1::3], paths[2::3]))


def read_image_from_file(file_path: str) -> np.ndarray:
    """Read image data from a file and return as is."""
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def save_image_to_file(img: np.ndarray, file_path: str) -> None:
    """Save an uint8 image to file, type is deduced from the extension."""
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_path, img)


def normalize_image(img: np.ndarray, img_size: Tuple[int, int] = None, fix_gamma: bool = False) -> np.ndarray:
    """Resize and convert to float if necessary. Return as RBG float32 image."""
    # convert uint8 images to float and normalize
    if img.dtype == np.uint8:
        img = img / 255.0

    # make sure the type is float32
    if img.dtype != np.float32:
        img = img.astype(np.float32)

    if fix_gamma:
        img = img ** 2.2

    # resize image if necessary
    if img_size is not None and img.shape[:2] != img_size:
        img = cv2.resize(img, img_size, interpolation=cv2.INTER_LINEAR)

    return img


def prepare_image_for_viewing(img: np.ndarray) -> np.ndarray:
    """Prepare an image for viewing/saving to a file."""
    img = np.clip(img, 0.0, 1.0)
    img = img ** (1.0 / 2.2)
    img = (img * 255.0) + 0.5
    img = img.astype(np.uint8)
    return img


def show_image(img: np.ndarray, grayscale: bool = False) -> None:
    """Plot a RGB image interactively."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots()
    axes.axis("off")

    img = prepare_image_for_viewing(img)

    if grayscale:
        axes.imshow(img, cmap="gray")
    else:
        axes.imshow(img)

    plt.show(fig)
    plt.close(fig)


def get_images(paths: Tuple[str, str, str], input_size: Tuple[int, int], fix_gamma: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read in images triplet, convert and resize if necessary, generate eroded mask, return RGB images."""
    input_img = normalize_image(read_image_from_file(paths[0]), input_size, fix_gamma)
    uv_img = normalize_image(read_image_from_file(paths[1]), input_size, fix_gamma)
    mask_img = normalize_image(read_image_from_file(paths[2]), input_size, fix_gamma)

    # threshold mask img as it is antialiased
    mask_img = ((mask_img == 1.0) * 1.0).astype(np.float32)
    uv_img = uv_img * mask_img

    return input_img, uv_img, mask_img


def augment_images(input_img: np.ndarray,
                   uv_img: np.ndarray,
                   mask_img: np.ndarray,
                   augment: AugmentParameters,
                   epoch_number: int,
                   occlusion_textures: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perform different augmentations on images."""
    mask_occluded_img = copy.deepcopy(mask_img)
    erode_kernel = np.ones((augment.mask_erosion_amount, augment.mask_erosion_amount), np.float32)
    mask_eroded_img = cv2.erode(mask_img, erode_kernel, iterations=1)
    mask_eroded_img = ((mask_eroded_img == 1.0) * 1.0).astype(np.float32)

    if augment.enabled:
        # this assumes float32 occlusion textures
        if augment.occlusion and random.uniform(0.0, 1.0) < augment.occlusion_prob:
            occlusion_texture = random.choice(occlusion_textures)
            color_texture = random.choice(occlusion_textures)
            occlusion = None

            for i in range(1000):
                occlusion_channel = occlusion_texture[:, :, random.randint(0, 2)]

                r1 = random.randint(0, occlusion_channel.shape[0] - input_img.shape[0] - 1)
                c1 = random.randint(0, occlusion_channel.shape[1] - input_img.shape[1] - 1)
                r2 = r1 + input_img.shape[0]
                c2 = c1 + input_img.shape[1]

                occlusion = occlusion_channel[r1:r2, c1:c2]
                occlusion = (1.0 - np.clip((occlusion - np.random.uniform() * 1.1) * np.random.uniform() * 50.0, 0.0, 1.0)).astype(np.float32)
                ratio = float(np.count_nonzero(occlusion)) / float(occlusion.size)

                # noinspection PyTypeChecker
                if 0.6 <= ratio <= 0.8:
                    break

            occlusion = np.rot90(occlusion, random.randint(0, 3))
            occlusion = occlusion[:, :, None]
            occlusion = np.repeat(occlusion, 3, axis=2)

            r1 = random.randint(0, color_texture.shape[0] - input_img.shape[0] - 1)
            c1 = random.randint(0, color_texture.shape[1] - input_img.shape[1] - 1)
            r2 = r1 + input_img.shape[0]
            c2 = c1 + input_img.shape[1]
            occlusion_color = color_texture[r1:r2, c1:c2]
            occlusion_color = np.rot90(occlusion_color, random.randint(0, 3))

            input_img = occlusion * input_img + (1.0 - occlusion) * occlusion_color
            mask_occluded_img = occlusion * mask_img

        if augment.shuffle and random.uniform(0.0, 1.0) < augment.shuffle_prob:
            channels = np.split(input_img, 3, axis=2)
            np.random.shuffle(channels)
            input_img = np.concatenate(channels, axis=2)

        if augment.rotate and random.uniform(0.0, 1.0) < augment.rotate_prob:
            k = random.randint(1, 3)
            input_img = np.rot90(input_img, k)
            uv_img = np.rot90(uv_img, k)
            mask_img = np.rot90(mask_img, k)
            mask_eroded_img = np.rot90(mask_eroded_img, k)
            mask_occluded_img = np.rot90(mask_occluded_img, k)

        if augment.exposure:
            input_img = input_img * random.uniform(augment.exposure_min, augment.exposure_max)

        if augment.gamma:
            input_img = np.clip(input_img, 0.0, np.finfo(np.float32).max)
            input_img = input_img ** random.uniform(augment.gamma_min, augment.gamma_max)

        if augment.noise:
            noise_scale_min = get_sched_value(augment.noise_scale_min, epoch_number)
            noise_scale_max = get_sched_value(augment.noise_scale_max, epoch_number)

            input_img = input_img + np.random.normal(loc=augment.noise_mean,
                                                     scale=augment.noise_std,
                                                     size=input_img.shape).astype(np.float32) * random.uniform(noise_scale_min, noise_scale_max)

        if augment.clip:
            input_img = np.clip(input_img, 0.0, 1.0)

        if augment.quantize:
            input_img = ((input_img * augment.quantization_levels + 0.5).astype(np.int32) / augment.quantization_levels).astype(np.float32)

    return input_img, uv_img, mask_img, mask_occluded_img, mask_eroded_img


def convert_image_to_data(img: np.ndarray, data_type: DataTypes) -> np.ndarray:
    """Convert an image to CNTK data format."""
    if data_type == DataTypes.Input:
        return np.ascontiguousarray(swap_to_front(img), dtype=np.float32)

    if data_type == DataTypes.Uv:
        # remove blue channel from uv image RGB -> RG
        img = img[:, :, 0:2]
        return np.ascontiguousarray(swap_to_front(img), dtype=np.float32)

    if data_type == DataTypes.Mask:
        # just take the first mask channel RGB -> R
        img = img[:, :, 0]
        return np.ascontiguousarray(img, dtype=np.float32)

    raise RuntimeError("Invalid data type given")


def convert_data_to_image(data: np.ndarray, data_type: DataTypes) -> np.ndarray:
    """Convert CNTK data array to a RGB image."""
    if data_type == DataTypes.Input:
        return swap_to_back(data)

    if data_type == DataTypes.Uv:
        data = swap_to_back(data)
        return np.concatenate([data, np.zeros((data.shape[0], data.shape[1], 1), dtype=np.float32)], axis=2)  # add one black channel

    if data_type == DataTypes.Mask:
        data = data[:, :, None]
        return np.repeat(data, 3, axis=2)  # repeat first channel three time

    raise RuntimeError("Invalid data type given")


def get_images_diff(first: np.ndarray, second: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Take difference of two RGB images and visualize with green (positive) and red (negative)"""
    first_avg = np.average(first, axis=2)
    second_avg = np.average(second, axis=2)
    diff = first_avg - second_avg
    red = np.where(diff < 0.0, diff, [0.0])
    green = np.where(diff >= 0.0, diff, [0.0])
    result = np.zeros(first.shape)
    result[:, :, 0] = np.absolute(red)
    result[:, :, 1] = green

    if scale != 1.0:
        result *= scale

    return result


def get_image_gradients(uv_img: np.ndarray, scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculates both x and y gradients for u and v channels. Returns four gradient images (ux, uy, vx, vy) and two magnitude images."""
    u = uv_img[:, :, 0]
    v = uv_img[:, :, 1]

    u1 = u[:, 1:]
    u2 = u[:, :-1]
    ux_grad = np.zeros(u.shape)
    ux_grad[:, :-1] = np.absolute(u1 - u2)

    u1 = u[1:, :]
    u2 = u[:-1, :]
    uy_grad = np.zeros(u.shape)
    uy_grad[:-1, :] = np.absolute(u1 - u2)

    v1 = v[:, 1:]
    v2 = v[:, :-1]
    vx_grad = np.zeros(v.shape)
    vx_grad[:, :-1] = np.absolute(v1 - v2)

    v1 = v[1:, :]
    v2 = v[:-1, :]
    vy_grad = np.zeros(v.shape)
    vy_grad[:-1, :] = np.absolute(v1 - v2)

    u_grad_mag = np.sqrt(ux_grad ** 2 + uy_grad ** 2)
    v_grad_mag = np.sqrt(vx_grad ** 2 + vy_grad ** 2)

    if scale != 1.0:
        u_grad_mag *= scale
        v_grad_mag *= scale

    ux_grad = ux_grad[:, :, None]
    ux_grad = np.repeat(ux_grad, 3, axis=2)

    uy_grad = uy_grad[:, :, None]
    uy_grad = np.repeat(uy_grad, 3, axis=2)

    vx_grad = vx_grad[:, :, None]
    vx_grad = np.repeat(vx_grad, 3, axis=2)

    vy_grad = vy_grad[:, :, None]
    vy_grad = np.repeat(vy_grad, 3, axis=2)

    u_grad_mag = u_grad_mag[:, :, None]
    u_grad_mag = np.repeat(u_grad_mag, 3, axis=2)

    v_grad_mag = v_grad_mag[:, :, None]
    v_grad_mag = np.repeat(v_grad_mag, 3, axis=2)

    return ux_grad, uy_grad, vx_grad, vy_grad, u_grad_mag, v_grad_mag


def get_image_gradients_cntk(uv_img_data: cntk.Variable) -> Tuple[cntk.Function, cntk.Function, cntk.Function, cntk.Function]:
    """Calculates both x and y gradients for u and v channels in CNTK. Returns four gradients (ux, uy, vx, vy)."""
    u = uv_img_data[0, :, :]
    v = uv_img_data[1, :, :]

    ux1 = u[:, :, 1:]
    ux2 = u[:, :, :-1]
    ux_grad = cntk.ops.abs(ux1 - ux2)
    ux_grad = cntk.splice(ux_grad, cntk.constant(0.0), axis=2)

    uy1 = u[:, 1:, :]
    uy2 = u[:, :-1, :]
    uy_grad = cntk.ops.abs(uy1 - uy2)
    uy_grad = cntk.splice(uy_grad, cntk.constant(0.0), axis=1)

    vx1 = v[:, :, 1:]
    vx2 = v[:, :, :-1]
    vx_grad = cntk.ops.abs(vx1 - vx2)
    vx_grad = cntk.splice(vx_grad, cntk.constant(0.0), axis=2)

    vy1 = v[:, 1:, :]
    vy2 = v[:, :-1, :]
    vy_grad = cntk.ops.abs(vy1 - vy2)
    vy_grad = cntk.splice(vy_grad, cntk.constant(0.0), axis=1)

    return ux_grad, uy_grad, vx_grad, vy_grad


def get_texture_projection(texture_img: np.ndarray, uv_img: np.ndarray, mask_img: np.ndarray, filter_scale: int = 2) -> np.ndarray:
    """Replace pixels in the uv image with corresponding pixels from a texture."""
    texture_width = texture_img.shape[1]
    texture_height = texture_img.shape[0]

    result_width = uv_img.shape[1]
    result_height = uv_img.shape[0]
    scaled_width = result_width * filter_scale
    scaled_height = result_height * filter_scale

    uv_img_scaled = cv2.resize(uv_img, (scaled_height, scaled_width), interpolation=cv2.INTER_LINEAR)
    mask_img_scaled = cv2.resize(mask_img, (scaled_height, scaled_width), interpolation=cv2.INTER_LINEAR)

    result_img_scaled = np.zeros((scaled_height, scaled_width, 3), dtype=np.float32)

    for y in range(scaled_height):
        for x in range(scaled_width):
            uv = uv_img_scaled[y, x]
            mask = mask_img_scaled[y, x]

            if uv[0] < 0.0 or uv[0] > 1.0 or uv[1] < 0.0 or uv[1] > 1.0:
                continue

            tx = uv[0] * (texture_width - 1)
            ty = (1.0 - uv[1]) * (texture_height - 1)
            ix = int(math.floor(tx))
            iy = int(math.floor(ty))

            result_img_scaled[y, x] = texture_img[iy, ix] * mask

    result_img = cv2.resize(result_img_scaled, (result_height, result_width), interpolation=cv2.INTER_LINEAR)
    return result_img


def get_texture_projection_input(texture_img: np.ndarray,
                                 input_img: np.ndarray,
                                 uv_img: np.ndarray,
                                 mask_img: np.ndarray,
                                 filter_scale: int = 2,
                                 opacity: float = 1.0) -> np.ndarray:
    """Replace pixels in the input image with corresponding pixels from a texture."""
    texture_width = texture_img.shape[1]
    texture_height = texture_img.shape[0]

    result_width = input_img.shape[1]
    result_height = input_img.shape[0]
    scaled_width = result_width * filter_scale
    scaled_height = result_height * filter_scale

    input_img_scaled = cv2.resize(input_img, (scaled_height, scaled_width), interpolation=cv2.INTER_LINEAR)
    uv_img_scaled = cv2.resize(uv_img, (scaled_height, scaled_width), interpolation=cv2.INTER_LINEAR)
    mask_img_scaled = cv2.resize(mask_img, (scaled_height, scaled_width), interpolation=cv2.INTER_LINEAR)

    result_img_scaled = np.zeros((scaled_height, scaled_width, 3), dtype=np.float32)

    for y in range(scaled_height):
        for x in range(scaled_width):
            uv = np.clip(uv_img_scaled[y, x], 0.0, 1.0)
            mask = mask_img_scaled[y, x]

            tx = uv[0] * (texture_width - 1)
            ty = (1.0 - uv[1]) * (texture_height - 1)
            ix = int(math.floor(tx))
            iy = int(math.floor(ty))

            result_img_scaled[y, x] = input_img_scaled[y, x] * (1.0 - (mask * opacity)) + texture_img[iy, ix] * (mask * opacity)

    result_img = cv2.resize(result_img_scaled, (result_height, result_width), interpolation=cv2.INTER_LINEAR)
    return result_img


def get_texture_projection_inv(input_img: np.ndarray, uv_img: np.ndarray, mask_img: np.ndarray) -> np.ndarray:
    """Project the input image to a new image using given uv map and bilinear filtering."""
    image_width = input_img.shape[1]
    image_height = input_img.shape[0]

    color_weight = np.zeros((image_height, image_width, 3), dtype=np.float32)
    weight = np.zeros((image_height, image_width, 1), dtype=np.float32)

    for y in range(image_height):
        for x in range(image_width):
            color = input_img[y, x]
            uv = uv_img[y, x]
            mask = mask_img[y, x]

            if (mask <= 0.95).any():
                continue

            if uv[0] < 0.0 or uv[0] > 1.0 or uv[1] < 0.0 or uv[1] > 1.0:
                continue

            tx = uv[0] * image_width
            ty = (1.0 - uv[1]) * image_height
            ix = int(math.floor(tx))
            iy = int(math.floor(ty))
            fx = tx - float(ix)
            fy = ty - float(iy)

            if ix == image_width or iy == image_height:
                continue

            w = (1.0 - fx) * (1.0 - fy)
            color_weight[iy, ix] += color * w
            weight[iy, ix] += w

            if ix < image_width - 1:
                w = fx * (1.0 - fy)
                color_weight[iy, ix + 1] += color * w
                weight[iy, ix + 1] += w

            if iy < image_height - 1:
                w = (1.0 - fx) * fy
                color_weight[iy + 1, ix] += color * w
                weight[iy + 1, ix] += w

            if ix < image_width - 1 and iy < image_height - 1:
                w = fx * fy
                color_weight[iy + 1, ix + 1] += color * w
                weight[iy + 1, ix + 1] += w

    weight = np.where(weight == 0.0, [1.0], weight)
    result = color_weight / weight

    return result


def get_grid_lines_projection(input_img: np.ndarray,
                              uv_img: np.ndarray,
                              mask_img: np.ndarray,
                              filter_scale: int = 2,
                              grid_density: float = 30.0,
                              grid_width: float = 0.1,
                              grid_opacity: float = 0.5,
                              grid_color: np.ndarray = np.array([0.0, 1.0, 0.0])) -> np.ndarray:
    """Project procedural grid lines onto the input image using rescaled images in the intermediate steps."""
    result_width = input_img.shape[1]
    result_height = input_img.shape[0]
    scaled_width = result_width * filter_scale
    scaled_height = result_height * filter_scale

    input_img_scaled = cv2.resize(input_img, (scaled_height, scaled_width), interpolation=cv2.INTER_LINEAR)
    uv_img_scaled = cv2.resize(uv_img, (scaled_height, scaled_width), interpolation=cv2.INTER_LINEAR)
    mask_img_scaled = cv2.resize(mask_img, (scaled_height, scaled_width), interpolation=cv2.INTER_LINEAR)
    result_img_scaled = np.zeros((scaled_height, scaled_width, 3), dtype=np.float32)

    for sy in range(scaled_height):
        for sx in range(scaled_width):
            uv = uv_img_scaled[sy, sx]
            mask = mask_img_scaled[sy, sx]

            if uv[0] < 0.0 or uv[0] > 1.0 or uv[1] < 0.0 or uv[1] > 1.0:
                continue

            # create the grid procedurally
            x = math.modf(uv[0] * grid_density)[0]
            y = math.modf(uv[1] * grid_density)[0]

            if x < grid_width or y < grid_width or x > (1.0 - grid_width) or y > (1.0 - grid_width):
                input_color = input_img_scaled[sy, sx]
                result_img_scaled[sy, sx] = input_color * (1.0 - (mask * grid_opacity)) + grid_color * (mask * grid_opacity)
            else:
                result_img_scaled[sy, sx] = input_img_scaled[sy, sx]

    result_img = cv2.resize(result_img_scaled, (result_height, result_width), interpolation=cv2.INTER_LINEAR)
    return result_img


def test_point_in_triangle(p, p0, p1, p2):
    """Test if a point lies in a triangle and return barycentric coordinates."""
    u = p0[1] * p2[0] - p0[0] * p2[1] + (p2[1] - p0[1]) * p[0] + (p0[0] - p2[0]) * p[1]
    v = p0[0] * p1[1] - p0[1] * p1[0] + (p0[1] - p1[1]) * p[0] + (p1[0] - p0[0]) * p[1]

    if (u < 0) != (v < 0):
        return False, 0.0, 0.0, 0.0

    a = -p1[1] * p2[0] + p0[1] * (p2[0] - p1[0]) + p0[0] * (p1[1] - p2[1]) + p1[0] * p2[1]

    if a < 0.0:
        u = -u
        v = -v
        a = -a

    result = u > 0 and v > 0 and (u + v) <= a
    w = 1.0 - u - v

    return result, u, v, w


def test_points_in_triangle(p0, p1, p2, density):
    """Create an integer grid system procedurally and test if the points lie inside the triangle."""
    p0 = p0 * density
    p1 = p1 * density
    p2 = p2 * density
    min_x = math.floor(min(p0[0], min(p1[0], p2[0])))
    max_x = math.ceil(max(p0[0], max(p1[0], p2[0])))
    min_y = math.floor(min(p0[1], min(p1[1], p2[1])))
    max_y = math.ceil(max(p0[1], max(p1[1], p2[1])))
    y = min_y
    results = []

    while y <= max_y:
        x = min_x

        while x <= max_x:
            result, u, v, w = test_point_in_triangle([x, y], p0, p1, p2)

            if result:
                results.append((u, v, w))

            x += 1

        y += 1

    return results


def get_grid_points_projection(input_img: np.ndarray,
                               uv_img: np.ndarray,
                               mask_img: np.ndarray,
                               grid_density: float = 50.0,
                               grid_opacity: float = 0.8,
                               grid_color: np.ndarray = np.array([1.0, 1.0, 1.0], dtype=np.float32)) -> np.ndarray:
    """Project a procedural point grid onto the input image using barycentric coordinates and simple filtering."""
    result_width = input_img.shape[1]
    result_height = input_img.shape[0]

    grid = np.zeros((result_height, result_width, 3), dtype=np.float32)

    for sy in range(result_height - 1):
        for sx in range(result_width - 1):
            uv1 = uv_img[sy, sx]
            uv2 = uv_img[sy, sx + 1]
            uv3 = uv_img[sy + 1, sx]

            results = test_points_in_triangle(uv1, uv2, uv3, grid_density)

            for result in results:
                u, v, w = result
                grid[sy, sx] += u
                grid[sy, sx + 1] += v
                grid[sy + 1, sx] += w

            uv1 = uv_img[sy, sx + 1]
            uv2 = uv_img[sy + 1, sx + 1]
            uv3 = uv_img[sy + 1, sx]

            results = test_points_in_triangle(uv1, uv2, uv3, grid_density)

            for result in results:
                u, v, w = result
                grid[sy, sx + 1] += u
                grid[sy + 1, sx + 1] += v
                grid[sy + 1, sx] += w

    grid = np.clip(grid, 0.0, 1.0) * mask_img
    result_img = input_img * (1.0 - (grid * grid_opacity)) + (grid * grid_opacity) * grid_color

    return result_img


def insert_plot_image(row: int,
                      column: int,
                      title: str,
                      shape: Tuple[int, int],
                      outer_pad: int,
                      row_pad: int,
                      column_pad: int,
                      title_offset: int,
                      src_img: np.ndarray,
                      dst_img: Image,
                      dst_draw: ImageDraw,
                      dst_font: ImageFont) -> None:
    """Insert numpy image array inside a PIL image."""
    r = outer_pad + (row - 1) * row_pad + (row - 1) * shape[0]
    c = outer_pad + (column - 1) * column_pad + (column - 1) * shape[1]
    src_img = prepare_image_for_viewing(src_img)
    src_img = Image.fromarray(src_img, "RGB")
    dst_img.paste(src_img, (c, r))
    dst_draw.text((c, r + title_offset), title, font=dst_font, fill=(0, 0, 0))
    dst_draw.text((c, r + title_offset), title, font=dst_font, fill=(0, 0, 0))  # draw text twice


def create_test_image_plot(input_orig_img: np.ndarray,
                           input_aug_img: np.ndarray,
                           target_uv_img: np.ndarray,
                           target_uv_mod_img: np.ndarray,
                           target_uv_grad_u_mag_img: np.ndarray,
                           target_uv_grad_v_mag_img: np.ndarray,
                           target_mask_img: np.ndarray,
                           target_mask_occluded_img: np.ndarray,
                           target_uv_proj1_img: np.ndarray,
                           target_uv_proj2_img: np.ndarray,
                           input_masked_img: np.ndarray,
                           result_uv_img: np.ndarray,
                           result_uv_masked_img: np.ndarray,
                           result_uv_masked_mod_img: np.ndarray,
                           result_uv_grad_u_mag_img: np.ndarray,
                           result_uv_grad_v_mag_img: np.ndarray,
                           result_mask_img: np.ndarray,
                           result_mask_occluded_img: np.ndarray,
                           result_uv_proj1_img: np.ndarray,
                           result_uv_proj2_img: np.ndarray,
                           target_mask_eroded_img: np.ndarray,
                           uv_diff_img: np.ndarray,
                           uv_mod_diff_img: np.ndarray,
                           grad_u_mag_diff_img: np.ndarray,
                           grad_v_mag_diff_img: np.ndarray,
                           mask_diff_img: np.ndarray,
                           mask_occluded_diff_img: np.ndarray,
                           uv_proj1_diff_img: np.ndarray,
                           uv_proj2_diff_img: np.ndarray,
                           epoch_number: int,
                           plot_number: int,
                           git_commit_name: str,
                           font_path: str,
                           output_path: str) -> None:
    """Create a plot image from given RGB images and save to a file."""
    shape = input_orig_img.shape
    rows = 3
    columns = 10
    outer_pad = 40
    row_pad = 40
    column_pad = 20
    title_offset = -18
    plot_img = Image.new("RGB", (2 * outer_pad + (columns - 1) * column_pad + columns * shape[1], 2 * outer_pad + (rows - 1) * row_pad + rows * shape[0]), (255, 255, 255))
    plot_draw = ImageDraw.Draw(plot_img)
    plot_font = ImageFont.truetype(font_path, 12)

    def insert_plot_image_(row, column, title, src_img):
        """Helper for insert image."""
        insert_plot_image(row, column, title, shape, outer_pad, row_pad, column_pad, title_offset, src_img, plot_img, plot_draw, plot_font)

    insert_plot_image_(1, 1, "Input orig", input_orig_img)
    insert_plot_image_(1, 2, "Input aug", input_aug_img)
    insert_plot_image_(1, 3, "Target uv", target_uv_img)
    insert_plot_image_(1, 4, "Target uv mod", target_uv_mod_img)
    insert_plot_image_(1, 5, "Target mask", target_mask_img)
    insert_plot_image_(1, 6, "Target mask occl", target_mask_occluded_img)
    insert_plot_image_(1, 7, "Target uv proj1", target_uv_proj1_img)
    insert_plot_image_(1, 8, "Target uv proj2", target_uv_proj2_img)
    insert_plot_image_(1, 9, "Target uv grad u", target_uv_grad_u_mag_img)
    insert_plot_image_(1, 10, "Target uv grad v", target_uv_grad_v_mag_img)

    insert_plot_image_(2, 1, "Input orig masked", input_masked_img)
    insert_plot_image_(2, 2, "Result uv no mask", result_uv_img)
    insert_plot_image_(2, 3, "Result uv", result_uv_masked_img)
    insert_plot_image_(2, 4, "Result uv mod", result_uv_masked_mod_img)
    insert_plot_image_(2, 5, "Result mask", result_mask_img)
    insert_plot_image_(2, 6, "Result mask occl", result_mask_occluded_img)
    insert_plot_image_(2, 7, "Result uv proj1", result_uv_proj1_img)
    insert_plot_image_(2, 8, "Result uv proj2", result_uv_proj2_img)
    insert_plot_image_(2, 9, "Result uv grad u", result_uv_grad_u_mag_img)
    insert_plot_image_(2, 10, "Result uv grad v", result_uv_grad_v_mag_img)

    uv_metric = np.asscalar(np.sum(np.abs(target_uv_img - result_uv_masked_img)))
    uv_mod_metric = np.asscalar(np.sum(np.abs(target_uv_mod_img - result_uv_masked_mod_img)))
    mask_metric = np.asscalar(np.sum(np.abs(target_mask_img - result_mask_img)))
    mask_occluded_metric = np.asscalar(np.sum(np.abs(target_mask_occluded_img - result_mask_occluded_img)))
    uv_proj1_metric = np.asscalar(np.sum(np.abs(target_uv_proj1_img - result_uv_proj1_img)))
    uv_proj2_metric = np.asscalar(np.sum(np.abs(target_uv_proj2_img - result_uv_proj2_img)))
    grad_u_metric = np.asscalar(np.sum(np.abs(target_uv_grad_u_mag_img - result_uv_grad_u_mag_img)))
    grad_v_metric = np.asscalar(np.sum(np.abs(target_uv_grad_v_mag_img - result_uv_grad_v_mag_img)))

    insert_plot_image_(3, 2, "Target mask erod", target_mask_eroded_img)
    insert_plot_image_(3, 3, "Uv diff ({0:.1f})".format(uv_metric), uv_diff_img)
    insert_plot_image_(3, 4, "Uv mod diff ({0:.1f})".format(uv_mod_metric), uv_mod_diff_img)
    insert_plot_image_(3, 5, "Mask diff ({0:.1f})".format(mask_metric), mask_diff_img)
    insert_plot_image_(3, 6, "Mask diff ({0:.1f})".format(mask_occluded_metric), mask_occluded_diff_img)
    insert_plot_image_(3, 7, "Uv proj1 diff ({0:.1f})".format(uv_proj1_metric), uv_proj1_diff_img)
    insert_plot_image_(3, 8, "Uv proj2 diff ({0:.1f})".format(uv_proj2_metric), uv_proj2_diff_img)
    insert_plot_image_(3, 9, "Grad u diff ({0:.1f})".format(grad_u_metric), grad_u_mag_diff_img)
    insert_plot_image_(3, 10, "Grad v diff ({0:.1f})".format(grad_v_metric), grad_v_mag_diff_img)

    plot_draw.text((10, plot_img.size[1] - 20), "{0} / {1} / {2}".format(git_commit_name, plot_number, epoch_number), font=plot_font, fill=(0, 0, 0))
    plot_img.save(output_path, format="PNG", optimize=True)


def plot_test_result(input_orig_img: np.ndarray,
                     input_aug_img: np.ndarray,
                     target_uv_img: np.ndarray,
                     target_mask_img: np.ndarray,
                     target_mask_occluded_img: np.ndarray,
                     target_mask_eroded_img: np.ndarray,
                     result_uv_img: np.ndarray,
                     result_mask_img: np.ndarray,
                     result_mask_occluded_img: np.ndarray,
                     face_texture_img: np.ndarray,
                     epoch_number: int,
                     plot_number: int,
                     git_commit_name: str,
                     font_path: str,
                     plot_path: str) -> None:
    """Helper function for creating plot images from test results."""
    uv_mod_factor = 25.0

    result_uv_mod_img = np.mod(uv_mod_factor * result_uv_img, 1.0)
    target_uv_mod_img = np.mod(uv_mod_factor * target_uv_img, 1.0)

    input_masked_img = input_orig_img * result_mask_img
    result_uv_masked_img = result_uv_img * result_mask_img
    result_uv_masked_mod_img = result_uv_mod_img * result_mask_img

    target_uv_proj1_img = get_texture_projection(face_texture_img, target_uv_img, target_mask_img)
    result_uv_proj1_img = get_texture_projection(face_texture_img, result_uv_masked_img, result_mask_img)
    target_uv_proj2_img = get_texture_projection_inv(input_orig_img, target_uv_img, target_mask_img)
    result_uv_proj2_img = get_texture_projection_inv(input_orig_img, result_uv_masked_img, result_mask_img)

    _, _, _, _, target_uv_grad_u_mag_img, target_uv_grad_v_mag_img = get_image_gradients(target_uv_img, 2.0)
    _, _, _, _, result_uv_grad_u_mag_img, result_uv_grad_v_mag_img = get_image_gradients(result_uv_img, 2.0)

    target_uv_grad_u_mag_img = target_uv_grad_u_mag_img * target_mask_eroded_img
    target_uv_grad_v_mag_img = target_uv_grad_v_mag_img * target_mask_eroded_img

    result_uv_grad_u_mag_img = result_uv_grad_u_mag_img * target_mask_eroded_img
    result_uv_grad_v_mag_img = result_uv_grad_v_mag_img * target_mask_eroded_img

    uv_diff_img = get_images_diff(target_uv_img, result_uv_masked_img)
    uv_mod_diff_img = get_images_diff(target_uv_mod_img, result_uv_masked_mod_img)
    grad_u_mag_diff_img = get_images_diff(target_uv_grad_u_mag_img, result_uv_grad_u_mag_img)
    grad_v_mag_diff_img = get_images_diff(target_uv_grad_v_mag_img, result_uv_grad_v_mag_img)
    mask_diff_img = get_images_diff(target_mask_img, result_mask_img)
    mask_occluded_diff_img = get_images_diff(target_mask_occluded_img, result_mask_occluded_img)
    uv_proj1_diff_img = get_images_diff(target_uv_proj1_img, result_uv_proj1_img)
    uv_proj2_diff_img = get_images_diff(target_uv_proj2_img, result_uv_proj2_img)

    create_test_image_plot(input_orig_img,
                           input_aug_img,
                           target_uv_img,
                           target_uv_mod_img,
                           target_uv_grad_u_mag_img,
                           target_uv_grad_v_mag_img,
                           target_mask_img,
                           target_mask_occluded_img,
                           target_uv_proj1_img,
                           target_uv_proj2_img,
                           input_masked_img,
                           result_uv_img,
                           result_uv_masked_img,
                           result_uv_masked_mod_img,
                           result_uv_grad_u_mag_img,
                           result_uv_grad_v_mag_img,
                           result_mask_img,
                           result_mask_occluded_img,
                           result_uv_proj1_img,
                           result_uv_proj2_img,
                           target_mask_eroded_img,
                           uv_diff_img,
                           uv_mod_diff_img,
                           grad_u_mag_diff_img,
                           grad_v_mag_diff_img,
                           mask_diff_img,
                           mask_occluded_diff_img,
                           uv_proj1_diff_img,
                           uv_proj2_diff_img,
                           epoch_number,
                           plot_number,
                           git_commit_name,
                           font_path,
                           os.path.join(plot_path, "{0:05d}_{1:02d}.png".format(epoch_number, plot_number)))


def create_real_image_plot(input_orig_img: np.ndarray,
                           input_masked_img: np.ndarray,
                           result_uv_img: np.ndarray,
                           result_uv_masked_img: np.ndarray,
                           result_uv_masked_mod_img: np.ndarray,
                           result_uv_grad_u_mag_img: np.ndarray,
                           result_uv_grad_v_mag_img: np.ndarray,
                           result_mask_img: np.ndarray,
                           result_mask_occluded_img: np.ndarray,
                           result_uv_proj1_img: np.ndarray,
                           result_uv_proj2_img: np.ndarray,
                           epoch_number: int,
                           plot_number: int,
                           git_commit_name: str,
                           font_path: str,
                           output_path: str):
    """Create a plot image from given RGB images and save to a file."""
    shape = input_orig_img.shape
    rows = 2
    columns = 5
    outer_pad = 40
    row_pad = 40
    column_pad = 20
    title_offset = -18
    plot_img = Image.new("RGB", (2 * outer_pad + (columns - 1) * column_pad + columns * shape[1], 2 * outer_pad + (rows - 1) * row_pad + rows * shape[0]), (255, 255, 255))
    plot_draw = ImageDraw.Draw(plot_img)
    plot_font = ImageFont.truetype(font_path, 12)

    def insert_plot_image_(row, column, title, src_img):
        """Helper for insert image."""
        insert_plot_image(row, column, title, shape, outer_pad, row_pad, column_pad, title_offset, src_img, plot_img, plot_draw, plot_font)

    insert_plot_image_(1, 1, "Input", input_orig_img)
    insert_plot_image_(1, 2, "Result uv", result_uv_masked_img)
    insert_plot_image_(1, 3, "Result mask", result_mask_img)
    insert_plot_image_(1, 4, "Result uv proj1", result_uv_proj1_img)
    insert_plot_image_(1, 5, "Result uv grad u", result_uv_grad_u_mag_img)

    insert_plot_image_(2, 1, "Input masked", input_masked_img)
    insert_plot_image_(2, 2, "Result uv mod", result_uv_masked_mod_img)
    insert_plot_image_(2, 3, "Result mask occl", result_mask_occluded_img)
    insert_plot_image_(2, 4, "Result uv proj2", result_uv_proj2_img)
    insert_plot_image_(2, 5, "Result uv grad v", result_uv_grad_v_mag_img)

    del result_uv_img  # unused
    # insert_plot_image_(0, 0, "Result uv no mask", result_uv_img)

    plot_draw.text((10, plot_img.size[1] - 20), "{0} / {1} / {2}".format(git_commit_name, plot_number, epoch_number), font=plot_font, fill=(0, 0, 0))
    plot_img.save(output_path, format="PNG", optimize=True)


def plot_real_result(input_orig_img: np.ndarray,
                     result_uv_img: np.ndarray,
                     result_mask_img: np.ndarray,
                     result_mask_occluded_img: np.ndarray,
                     face_texture_img: np.ndarray,
                     epoch_number: int,
                     plot_number: int,
                     git_commit_name: str,
                     font_path: str,
                     plot_path: str) -> None:
    """Helper function for creating plot images from real results."""
    uv_mod_factor = 25.0

    result_uv_mod_img = np.mod(uv_mod_factor * result_uv_img, 1.0)

    input_masked_img = input_orig_img * result_mask_img
    result_uv_masked_img = result_uv_img * result_mask_img
    result_uv_masked_mod_img = result_uv_mod_img * result_mask_img

    result_uv_proj1_img = get_texture_projection(face_texture_img, result_uv_masked_img, result_mask_img)
    result_uv_proj2_img = get_texture_projection_inv(input_orig_img, result_uv_masked_img, result_mask_img)

    _, _, _, _, result_uv_grad_u_mag_img, result_uv_grad_v_mag_img = get_image_gradients(result_uv_img, 2.0)

    result_uv_grad_u_mag_img = result_uv_grad_u_mag_img * result_mask_img
    result_uv_grad_v_mag_img = result_uv_grad_v_mag_img * result_mask_img

    create_real_image_plot(input_orig_img,
                           input_masked_img,
                           result_uv_img,
                           result_uv_masked_img,
                           result_uv_masked_mod_img,
                           result_uv_grad_u_mag_img,
                           result_uv_grad_v_mag_img,
                           result_mask_img,
                           result_mask_occluded_img,
                           result_uv_proj1_img,
                           result_uv_proj2_img,
                           epoch_number,
                           plot_number,
                           git_commit_name,
                           font_path,
                           os.path.join(plot_path, "{0:05d}_{1:02d}.png".format(epoch_number, plot_number)))


def create_real_image_plot2(input_img: np.ndarray,
                            grid_lines_proj_img: np.ndarray,
                            grid_points_proj_img: np.ndarray,
                            input_masked_img: np.ndarray,
                            texture_proj_img: np.ndarray,
                            texture_proj_input_img: np.ndarray,
                            result_mask_img: np.ndarray,
                            result_mask_occl_img: np.ndarray,
                            result_uv_img: np.ndarray,
                            texture_proj_inv_img: np.ndarray,
                            texture_proj_inv_avg_img: np.ndarray,
                            result_uv_mod_img: np.ndarray,
                            font_path: str,
                            output_path: str) -> None:
    shape = input_img.shape
    rows = 3
    columns = 4
    outer_pad = 40
    row_pad = 40
    column_pad = 20
    title_offset = -18
    plot_img = Image.new("RGB", (2 * outer_pad + (columns - 1) * column_pad + columns * shape[1], 2 * outer_pad + (rows - 1) * row_pad + rows * shape[0]), (255, 255, 255))
    plot_draw = ImageDraw.Draw(plot_img)
    plot_font = ImageFont.truetype(font_path, 12)

    def insert_plot_image_(row, column, title, src_img):
        insert_plot_image(row, column, title, shape, outer_pad, row_pad, column_pad, title_offset, src_img, plot_img, plot_draw, plot_font)

    insert_plot_image_(1, 1, "Input masked", input_masked_img)
    insert_plot_image_(1, 2, "Texture proj", texture_proj_img)
    insert_plot_image_(1, 3, "Tex proj on input", texture_proj_input_img)
    insert_plot_image_(1, 4, "Tex proj inv", texture_proj_inv_img)

    insert_plot_image_(2, 1, "Grid lines proj", grid_lines_proj_img)
    insert_plot_image_(2, 2, "Input", input_img)
    insert_plot_image_(2, 3, "Grid points proj", grid_points_proj_img)
    insert_plot_image_(2, 4, "Tex proj inv avg", texture_proj_inv_avg_img)

    insert_plot_image_(3, 1, "Result mask", result_mask_img)
    insert_plot_image_(3, 2, "Result mask occl", result_mask_occl_img)
    insert_plot_image_(3, 3, "Result uv", result_uv_img)
    insert_plot_image_(3, 4, "Result uv mod", result_uv_mod_img)

    plot_img.save(output_path, format="PNG", optimize=True)


def plot_real_result2(input_img: np.ndarray,
                      result_uv_img: np.ndarray,
                      result_mask_img: np.ndarray,
                      result_mask_occl_img: np.ndarray,
                      texture_proj_inv_images: np.ndarray,
                      face_texture_img: np.ndarray,
                      plot_number: int,
                      font_path: str,
                      plot_path: str) -> np.ndarray:
    """Helper function for creating plot images from real results."""
    uv_mod_factor = 25.0

    result_uv_mod_img = np.mod(uv_mod_factor * result_uv_img, 1.0)

    input_masked_img = input_img * result_mask_img
    result_uv_masked_img = result_uv_img * result_mask_img
    result_uv_masked_mod_img = result_uv_mod_img * result_mask_img

    grid_lines_proj_img = get_grid_lines_projection(input_img, result_uv_img, result_mask_img)
    grid_points_proj_img = get_grid_points_projection(input_img, result_uv_img, result_mask_img)
    texture_proj_img = get_texture_projection(face_texture_img, result_uv_masked_img, result_mask_img)
    texture_proj_input_img = get_texture_projection_input(face_texture_img, input_img, result_uv_masked_img, result_mask_img)
    texture_proj_inv_img = get_texture_projection_inv(input_img, result_uv_masked_img, result_mask_img)
    texture_proj_inv_images[plot_number % texture_proj_inv_images.shape[0]] = texture_proj_inv_img
    texture_proj_inv_avg_img = np.sum(texture_proj_inv_images, axis=0) / texture_proj_inv_images.shape[0]

    create_real_image_plot2(input_img,
                            grid_lines_proj_img,
                            grid_points_proj_img,
                            input_masked_img,
                            texture_proj_img,
                            texture_proj_input_img,
                            result_mask_img,
                            result_mask_occl_img,
                            result_uv_masked_img,
                            texture_proj_inv_img,
                            texture_proj_inv_avg_img,
                            result_uv_masked_mod_img,
                            font_path,
                            os.path.join(plot_path, "{0:05d}.png".format(plot_number)))

    return texture_proj_inv_img


def plot_all_results(test_results: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
                     real_results: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
                     epoch_number: int,
                     git_commit_name: str,
                     font_path: str,
                     plot_path: str,
                     face_texture_path: str) -> None:
    """Plot all the results into image files."""
    face_texture_img = normalize_image(read_image_from_file(face_texture_path), fix_gamma=True)
    plot_number = 1

    for result in test_results:
        input_orig_img, input_aug_img, target_uv_img, target_mask_img, target_mask_occluded_img, \
        target_mask_eroded_img, result_uv_img, result_mask_img, result_mask_occluded_img = result

        plot_test_result(input_orig_img, input_aug_img, target_uv_img, target_mask_img, target_mask_occluded_img, target_mask_eroded_img,
                         result_uv_img, result_mask_img, result_mask_occluded_img, face_texture_img,
                         epoch_number, plot_number, git_commit_name, font_path, plot_path)

        plot_number += 1

    for result in real_results:
        input_orig_img, result_uv_img, result_mask_img, result_mask_occluded_img = result

        plot_real_result(input_orig_img, result_uv_img, result_mask_img, result_mask_occluded_img, face_texture_img,
                         epoch_number, plot_number, git_commit_name, font_path, plot_path)

        plot_number += 1


def evaluate_model_single(model: cntk.Function, input_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the model with input image and return uv and masks images."""
    input_image_data = convert_image_to_data(input_img, DataTypes.Input)

    result_data = model.eval([input_image_data])[0].squeeze()
    result_uv_data = result_data[0:2, :, :]
    result_mask_data = result_data[2, :, :]
    result_mask_occluded_data = result_data[3, :, :]

    result_uv_img = convert_data_to_image(result_uv_data, DataTypes.Uv)
    result_mask_img = convert_data_to_image(result_mask_data, DataTypes.Mask)
    result_mask_occluded_img = convert_data_to_image(result_mask_occluded_data, DataTypes.Mask)

    return result_uv_img, result_mask_img, result_mask_occluded_img


def generate_random_augment():
    """Generate different augments with fixed probabilities."""
    augment = AugmentParameters()
    augment.enabled = False  # 1/3 of the time disabled
    value = random.uniform(0.0, 1.0)

    # noinspection PyTypeChecker
    if 0.333 <= value < 0.666:
        # only occlusion
        augment.enabled = True
        augment.occlusion = True
        augment.shuffle = False
        augment.rotate = False
        augment.exposure = False
        augment.gamma = False
        augment.noise = False
        augment.clip = True
        augment.quantize = True
    elif 0.666 <= value <= 1.0:
        # everything but rotation
        augment.enabled = True
        augment.occlusion = True
        augment.shuffle = True
        augment.rotate = False
        augment.exposure = True
        augment.gamma = True
        augment.noise = True
        augment.clip = True
        augment.quantize = True

    return augment


def evaluate_model_multiple(model: cntk.Function,
                            params: Parameters,
                            eval_test_image_paths: List[Tuple[str, str, str]],
                            eval_real_image_aligned_paths: List[str],
                            eval_real_image_paths: List[str],
                            occlusion_textures: List[np.ndarray],
                            epoch_number: int) -> Tuple[List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
                                                        List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
    """Evaluate the model with test and real images. Return the output as lists of image tuples."""
    test_results = []
    real_results = []

    for paths in eval_test_image_paths:
        input_orig_img, target_uv_img, target_mask_img = get_images(paths, params.input_size)

        if params.use_random_eval_augment:
            augment = generate_random_augment()
        else:
            augment = params.eval_augment

        input_aug_img, target_uv_img, target_mask_img, \
        target_mask_occluded_img, target_mask_eroded_img = augment_images(input_orig_img, target_uv_img, target_mask_img, augment, epoch_number, occlusion_textures)

        result_uv_img, result_mask_img, result_mask_occluded_img = evaluate_model_single(model, input_aug_img)

        test_results.append((input_orig_img, input_aug_img, target_uv_img, target_mask_img, target_mask_occluded_img,
                             target_mask_eroded_img, result_uv_img, result_mask_img, result_mask_occluded_img))

    for path in eval_real_image_aligned_paths:
        input_orig_img = read_image_from_file(path)
        input_orig_img = input_orig_img[20:198, :, :]  # slice top and bottom off
        input_orig_img = normalize_image(input_orig_img, params.input_size, True)
        result_uv_img, result_mask_img, result_mask_occluded_img = evaluate_model_single(model, input_orig_img)
        real_results.append((input_orig_img, result_uv_img, result_mask_img, result_mask_occluded_img))

    for path in eval_real_image_paths:
        input_orig_img = normalize_image(read_image_from_file(path), params.input_size, True)
        result_uv_img, result_mask_img, result_mask_occluded_img = evaluate_model_single(model, input_orig_img)
        real_results.append((input_orig_img, result_uv_img, result_mask_img, result_mask_occluded_img))

    return test_results, real_results


def create_html_template(template_path: str) -> jinja2.Template:
    """Read in the html template file and create a jinja2 template."""
    with open(template_path) as f:
        return jinja2.Template(f.read())


def write_html(template: jinja2.Template,
               html_path: str,
               params: Parameters,
               run_id: str,
               git_commit_name: str,
               epoch_count: int,
               elapsed_time: str,
               param_count: int,
               samples_seen: int,
               samples_per_s: float,
               train_losses: List[float],
               test_losses: List[float],
               eval_metrics: List[float]) -> None:
    """Write the jinja2 template to a file."""
    html_string = template.render(
        run_id=run_id,
        run_description=params.run_description,
        git_commit_name=git_commit_name,
        epoch_count=epoch_count,
        elapsed_time=elapsed_time,
        param_count=intspace(param_count),
        samples_seen=intspace(samples_seen),
        samples_per_s="{0:02.1f}".format(samples_per_s),
        params_str=params.to_string(),
        train_losses=str(train_losses),
        test_losses=str(test_losses),
        eval_metrics=str(eval_metrics)
    )

    with open(html_path, "w") as f:
        f.write(html_string)


def print_code_as_html(code_path: str, html_path: str) -> None:
    """Format given python code file to a html file."""
    with open(code_path) as infile:
        with open(html_path, "w") as outfile:
            pygments.highlight(infile.read(), PythonLexer(), HtmlFormatter(full=True), outfile)


def create_image_collage(images: List[np.ndarray], images_per_row: int, output_path: str) -> None:
    """Create a collage of float32 images, assumes all images are of the same size, saves to a PNG file."""
    images_count = len(images)
    row_count = int(math.ceil(images_count / images_per_row))
    column_count = images_per_row
    image_size = images[0].shape
    result_img = np.zeros((row_count * image_size[0], column_count * image_size[1], 3), dtype=np.float32)

    for r in range(row_count):
        for c in range(column_count):
            i = r * images_per_row + c

            if i >= images_count:
                continue

            r1 = r * image_size[0]
            r2 = r1 + image_size[0]
            c1 = c * image_size[1]
            c2 = c1 + image_size[1]

            result_img[r1:r2, c1:c2] = images[i]

    result_img = prepare_image_for_viewing(result_img)
    pil_img = Image.fromarray(result_img, "RGB")
    pil_img.save(output_path, format="PNG", optimize=True)
