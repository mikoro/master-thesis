# coding=utf-8

"""UV-net deep convolutional neural network training functions."""

# The target GPU can be selected with the --gpu switch for the script.
# If omitted, the best available GPU is automatically selected.
# Run parameters can be loaded from a file using the --params switch.

import argparse
import datetime
import multiprocessing as mp
import shutil
import signal
import time
import traceback

from uvnet_utils import *


# --------------------------------------------------------------------------------


def setup_parameters() -> Parameters:
    """Setup the training parameters."""
    params = Parameters()
    params.run_description = "baseline"

    if os.name == "nt":
        params.model_load_from_file = False
        params.model_load_file_name = "uvnet.model"
        params.output_directory = "E:/Thesis/results"
        params.textures_directory = "E:/Thesis/textures"
        params.train_image_directories = ["E:/Thesis/renders/head_final7_train"]
        params.test_image_directories = ["E:/Thesis/renders/head_final7_test"]
        params.real_image_aligned_directories = ["E:/Thesis/misc/real_celeba_aligned"]
        params.real_image_directories = ["E:/Thesis/misc/real_celeba"]
    elif os.name == "posix":
        tmp_dir, wrk_dir = os.environ["TMPDIR"], os.environ["WRKDIR"]
        assert len(tmp_dir) > 0 and len(wrk_dir) > 0
        params.model_load_from_file = True
        params.model_load_file_name = wrk_dir + "/uvnet_with_noise.model"
        # params.model_load_file_name = wrk_dir + "/uvnet_no_noise.model"
        params.output_directory = wrk_dir + "/results"
        params.textures_directory = wrk_dir + "/misc/textures"
        params.train_image_directories = [tmp_dir + "/head_final7_train"]
        params.test_image_directories = [tmp_dir + "/head_final7_test"]
        params.real_image_aligned_directories = [wrk_dir + "/misc/real_celeba_aligned"]
        params.real_image_directories = [wrk_dir + "/misc/real_celeba"]

    return params


# --------------------------------------------------------------------------------


def setup_cntk(params: Parameters) -> Tuple[cntk.Function, cntk.Function, cntk.Function, cntk.Learner, cntk.learning_rate_schedule, cntk.momentum_schedule,
                                            cntk.Variable, cntk.Variable, cntk.Variable, cntk.Variable, cntk.Variable]:
    """Setup CNTK model, loss function and learner."""
    # cntk input and target variables
    input_image = cntk.input_variable((3, params.input_size[0], params.input_size[1]))
    target_uv = cntk.input_variable((2, params.input_size[0], params.input_size[1]))
    target_mask = cntk.input_variable((1, params.input_size[0], params.input_size[1]))
    target_mask_occluded = cntk.input_variable((1, params.input_size[0], params.input_size[1]))
    target_mask_eroded = cntk.input_variable((1, params.input_size[0], params.input_size[1]))

    # create the model
    model = create_model(input_image, params)

    # loss function
    result_uv = model[0:2, :, :]
    result_mask = model[2, :, :]
    result_mask_occluded = model[3, :, :]

    target_ux_grad, target_uy_grad, target_vx_grad, target_vy_grad = get_image_gradients_cntk(target_uv)
    result_ux_grad, result_uy_grad, result_vx_grad, result_vy_grad = get_image_gradients_cntk(result_uv)

    target_ux_grad = target_ux_grad * target_mask_eroded
    target_uy_grad = target_uy_grad * target_mask_eroded
    target_vx_grad = target_vx_grad * target_mask_eroded
    target_vy_grad = target_vy_grad * target_mask_eroded

    result_ux_grad = result_ux_grad * target_mask_eroded
    result_uy_grad = result_uy_grad * target_mask_eroded
    result_vx_grad = result_vx_grad * target_mask_eroded
    result_vy_grad = result_vy_grad * target_mask_eroded

    result_uv_masked = result_uv * target_mask

    loss_function = \
        cntk_l1_loss(result_uv_masked, target_uv) + \
        cntk_l1_loss(result_mask, target_mask) + \
        cntk_l1_loss(result_mask_occluded, target_mask_occluded) + \
        cntk_l1_loss(result_ux_grad, target_ux_grad) + \
        cntk_l1_loss(result_uy_grad, target_uy_grad) + \
        cntk_l1_loss(result_vx_grad, target_vx_grad) + \
        cntk_l1_loss(result_vy_grad, target_vy_grad)

    eval_function = cntk_l1_loss(result_uv_masked, target_uv)

    # learning rate and momentum schedules
    epoch_size = params.train_epoch_size if isinstance(params.learner.learning_rate, list) else None
    learning_rate_schedule = cntk.learning_rate_schedule(params.learner.learning_rate, unit=cntk.UnitType.sample, epoch_size=epoch_size)
    epoch_size = params.train_epoch_size if isinstance(params.learner.momentum, list) else None
    momentum_schedule = cntk.momentum_schedule(params.learner.momentum, epoch_size=epoch_size)

    cntk.learners.set_default_use_mean_gradient_value(params.learner.use_mean_gradient)

    learner = cntk.adam(model.parameters, lr=learning_rate_schedule, momentum=momentum_schedule,
                        l1_regularization_weight=params.learner.l1_regularization_weight,
                        l2_regularization_weight=params.learner.l2_regularization_weight,
                        gaussian_noise_injection_std_dev=params.learner.gaussian_noise_injection_std_dev,
                        gradient_clipping_threshold_per_sample=params.learner.gradient_clipping_threshold_per_sample,
                        gradient_clipping_with_truncation=params.learner.gradient_clipping_with_truncation)

    return model, loss_function, eval_function, learner, learning_rate_schedule, momentum_schedule, \
           input_image, target_uv, target_mask, target_mask_occluded, target_mask_eroded


# --------------------------------------------------------------------------------


def create_model(input_image: cntk.Variable, params: Parameters) -> cntk.Function:
    """Create the network model"""
    cnf = [int(round(params.model.initial_features * pow(params.model.feature_multiplier, i))) for i in range(0, 7)]  # convolution num filters
    ucnf = [int(round(params.model.up_factor * i)) for i in cnf]  # up convolution num filters
    fs = params.model.filter_size

    with cntk.default_options(init=cntk.glorot_uniform(), activation=cntk.relu, pad=True, bias=True):
        p1, p2, p3, p4, p5, p6 = None, None, None, None, None, None

        # downsample

        l = cntk.layers.Convolution(filter_shape=fs, strides=(1, 1), num_filters=cnf[0])(input_image)  # Nx128x128
        l = cntk.layers.Convolution(filter_shape=fs, strides=(1, 1), num_filters=cnf[0])(l)  # Nx128x128
        p1 = l = cntk.layers.MaxPooling(filter_shape=(2, 2), strides=(2, 2))(l)  # Nx64x64

        if params.model.levels >= 2:
            l = cntk.layers.Convolution(filter_shape=fs, strides=(1, 1), num_filters=cnf[1])(l)  # 64x64
            l = cntk.layers.Convolution(filter_shape=fs, strides=(1, 1), num_filters=cnf[1])(l)  # 64x64
            p2 = l = cntk.layers.MaxPooling(filter_shape=(2, 2), strides=(2, 2))(l)  # 32x32

        if params.model.levels >= 3:
            l = cntk.layers.Convolution(filter_shape=fs, strides=(1, 1), num_filters=cnf[2])(l)  # 32x32
            l = cntk.layers.Convolution(filter_shape=fs, strides=(1, 1), num_filters=cnf[2])(l)  # 32x32
            p3 = l = cntk.layers.MaxPooling(filter_shape=(2, 2), strides=(2, 2))(l)  # 16x16

        if params.model.levels >= 4:
            l = cntk.layers.Convolution(filter_shape=fs, strides=(1, 1), num_filters=cnf[3])(l)  # 16x16
            l = cntk.layers.Convolution(filter_shape=fs, strides=(1, 1), num_filters=cnf[3])(l)  # 16x16
            p4 = l = cntk.layers.MaxPooling(filter_shape=(2, 2), strides=(2, 2))(l)  # 8x8

        if params.model.levels >= 5:
            l = cntk.layers.Convolution(filter_shape=fs, strides=(1, 1), num_filters=cnf[4])(l)  # 8x8
            l = cntk.layers.Convolution(filter_shape=fs, strides=(1, 1), num_filters=cnf[4])(l)  # 8x8
            p5 = l = cntk.layers.MaxPooling(filter_shape=(2, 2), strides=(2, 2))(l)  # 4x4

        if params.model.levels >= 6:
            l = cntk.layers.Convolution(filter_shape=(3, 3), strides=(1, 1), num_filters=cnf[5])(l)  # 4x4, fixed filter shape
            l = cntk.layers.Convolution(filter_shape=(3, 3), strides=(1, 1), num_filters=cnf[5])(l)  # 4x4
            p6 = l = cntk.layers.MaxPooling(filter_shape=(2, 2), strides=(2, 2))(l)  # 2x2

        if params.model.levels >= 7:
            l = cntk.layers.Convolution(filter_shape=(1, 1), strides=(1, 1), num_filters=cnf[6])(l)  # 2x2, (2, 2) filter shape causes issues with cuDNN 6.0
            l = cntk.layers.Convolution(filter_shape=(1, 1), strides=(1, 1), num_filters=cnf[6])(l)  # 2x2
            l = cntk.layers.MaxPooling(filter_shape=(2, 2), strides=(2, 2))(l)  # 1x1

        # upsample

        if params.model.levels >= 7:
            l = cntk.layers.ConvolutionTranspose(filter_shape=(2, 2), strides=(2, 2), num_filters=ucnf[6], output_shape=(2, 2))(l)
            l = cntk.layers.Convolution(filter_shape=(1, 1), strides=(1, 1), num_filters=ucnf[6])(l)  # (2, 2) filter shape causes issues with cuDNN 6.0
            l = cntk.layers.Convolution(filter_shape=(1, 1), strides=(1, 1), num_filters=ucnf[6])(l)
            l = cntk.ops.splice(l, p6, axis=0)

        if params.model.levels >= 6:
            l = cntk.layers.ConvolutionTranspose(filter_shape=(3, 3), strides=(2, 2), num_filters=ucnf[5], output_shape=(4, 4))(l)
            l = cntk.layers.Convolution(filter_shape=(3, 3), strides=(1, 1), num_filters=ucnf[5])(l)  # fixed filter shape
            l = cntk.layers.Convolution(filter_shape=(3, 3), strides=(1, 1), num_filters=ucnf[5])(l)
            l = cntk.ops.splice(l, p5, axis=0)

        if params.model.levels >= 5:
            l = cntk.layers.ConvolutionTranspose(filter_shape=fs, strides=(2, 2), num_filters=ucnf[4], output_shape=(8, 8))(l)
            l = cntk.layers.Convolution(filter_shape=fs, strides=(1, 1), num_filters=ucnf[4])(l)
            l = cntk.layers.Convolution(filter_shape=fs, strides=(1, 1), num_filters=ucnf[4])(l)
            l = cntk.ops.splice(l, p4, axis=0)

        if params.model.levels >= 4:
            l = cntk.layers.ConvolutionTranspose(filter_shape=fs, strides=(2, 2), num_filters=ucnf[3], output_shape=(16, 16))(l)
            l = cntk.layers.Convolution(filter_shape=fs, strides=(1, 1), num_filters=ucnf[3])(l)
            l = cntk.layers.Convolution(filter_shape=fs, strides=(1, 1), num_filters=ucnf[3])(l)
            l = cntk.ops.splice(l, p3, axis=0)

        if params.model.levels >= 3:
            l = cntk.layers.ConvolutionTranspose(filter_shape=fs, strides=(2, 2), num_filters=ucnf[2], output_shape=(32, 32))(l)
            l = cntk.layers.Convolution(filter_shape=fs, strides=(1, 1), num_filters=ucnf[2])(l)
            l = cntk.layers.Convolution(filter_shape=fs, strides=(1, 1), num_filters=ucnf[2])(l)
            l = cntk.ops.splice(l, p2, axis=0)

        if params.model.levels >= 2:
            l = cntk.layers.ConvolutionTranspose(filter_shape=fs, strides=(2, 2), num_filters=ucnf[1], output_shape=(64, 64))(l)
            l = cntk.layers.Convolution(filter_shape=fs, strides=(1, 1), num_filters=ucnf[1])(l)
            l = cntk.layers.Convolution(filter_shape=fs, strides=(1, 1), num_filters=ucnf[1])(l)
            l = cntk.ops.splice(l, p1, axis=0)

        l = cntk.layers.ConvolutionTranspose(filter_shape=fs, strides=(2, 2), num_filters=ucnf[0], output_shape=(128, 128))(l)
        l = cntk.layers.Convolution(filter_shape=fs, strides=(1, 1), num_filters=ucnf[0])(l)
        l = cntk.layers.Convolution(filter_shape=fs, strides=(1, 1), num_filters=ucnf[0])(l)
        l = cntk.ops.splice(l, input_image, axis=0)

        l = cntk.layers.Convolution(filter_shape=fs, strides=(1, 1), num_filters=ucnf[0])(l)
        l = cntk.layers.Convolution(filter_shape=fs, strides=(1, 1), num_filters=ucnf[0])(l)
        l = cntk.layers.Convolution(filter_shape=fs, strides=(1, 1), num_filters=4, activation=None)(l)

    return l


# --------------------------------------------------------------------------------

interrupted = False


def signal_handler(signum: int, frame: int) -> None:
    """Function for handling the interrupt etc. signals."""
    del signum, frame  # unused
    global interrupted
    interrupted = True
    name = mp.current_process().name

    if name is not None:
        print("\nProcess \"{0}\" interrupted!\n".format(name))
    else:
        print("\nInterrupted!\n")


# --------------------------------------------------------------------------------


def read_data_process_func(params: Parameters,
                           epoch_number: mp.Value,
                           minibatch_pipe: mp.Pipe,
                           should_stop: mp.Event,
                           should_proceed: mp.Event,
                           should_send: mp.Event,
                           epoch_finished: mp.Event,
                           train_image_paths: List[Tuple[str, str, str]],
                           test_image_paths: List[Tuple[str, str, str]],
                           occlusion_texture_paths: List[str]) -> None:
    """Read minibatch data in a separate process and send it through a pipe."""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        occlusion_textures = []

        for occlusion_texture_path in occlusion_texture_paths:
            occlusion_textures.append(normalize_image(read_image_from_file(occlusion_texture_path)))

        while not should_stop.is_set() and not interrupted:
            image_index = 0
            random.shuffle(train_image_paths)

            while True:
                if should_stop.is_set() or interrupted:
                    break
                if should_proceed.wait(0.1):
                    should_proceed.clear()
                    break

            # train data reading loop
            while image_index < params.train_epoch_size and not should_stop.is_set() and not interrupted:
                minibatch_size = min(params.train_minibatch_size, params.train_epoch_size - image_index)
                input_image_minibatch = np.zeros((minibatch_size, 3, params.input_size[0], params.input_size[1]), dtype=np.float32)
                target_uv_minibatch = np.zeros((minibatch_size, 2, params.input_size[0], params.input_size[1]), dtype=np.float32)
                target_mask_minibatch = np.zeros((minibatch_size, 1, params.input_size[0], params.input_size[1]), dtype=np.float32)
                target_mask_occluded_minibatch = np.zeros((minibatch_size, 1, params.input_size[0], params.input_size[1]), dtype=np.float32)
                target_mask_eroded_minibatch = np.zeros((minibatch_size, 1, params.input_size[0], params.input_size[1]), dtype=np.float32)

                for i in range(0, minibatch_size):
                    paths = train_image_paths[image_index + i]

                    input_orig_img, target_uv_img, target_mask_img = get_images(paths, params.input_size)

                    input_aug_img, target_uv_img, target_mask_img, \
                    target_mask_occluded_img, target_mask_eroded_img = augment_images(input_orig_img, target_uv_img, target_mask_img,
                                                                                      params.train_augment, epoch_number.value, occlusion_textures)

                    input_image_data = convert_image_to_data(input_aug_img, DataTypes.Input)
                    target_uv_data = convert_image_to_data(target_uv_img, DataTypes.Uv)
                    target_mask_data = convert_image_to_data(target_mask_img, DataTypes.Mask)
                    target_mask_occluded_data = convert_image_to_data(target_mask_occluded_img, DataTypes.Mask)
                    target_mask_eroded_data = convert_image_to_data(target_mask_eroded_img, DataTypes.Mask)

                    input_image_minibatch[i] = input_image_data
                    target_uv_minibatch[i] = target_uv_data
                    target_mask_minibatch[i] = target_mask_data
                    target_mask_occluded_minibatch[i] = target_mask_occluded_data
                    target_mask_eroded_minibatch[i] = target_mask_eroded_data

                image_index += minibatch_size

                while True:
                    if should_stop.is_set() or interrupted:
                        break
                    if should_send.wait(0.1):
                        should_send.clear()
                        minibatch_pipe.send((input_image_minibatch, target_uv_minibatch, target_mask_minibatch, target_mask_occluded_minibatch, target_mask_eroded_minibatch))
                        break

            epoch_finished.set()

            image_index = 0
            random.shuffle(test_image_paths)

            while True:
                if should_stop.is_set() or interrupted:
                    break
                if should_proceed.wait(0.1):
                    should_proceed.clear()
                    break

            # test data reading loop
            while image_index < params.test_epoch_size and not should_stop.is_set() and not interrupted:
                minibatch_size = min(params.test_minibatch_size, params.test_epoch_size - image_index)
                input_image_minibatch = np.zeros((minibatch_size, 3, params.input_size[0], params.input_size[1]), dtype=np.float32)
                target_uv_minibatch = np.zeros((minibatch_size, 2, params.input_size[0], params.input_size[1]), dtype=np.float32)
                target_mask_minibatch = np.zeros((minibatch_size, 1, params.input_size[0], params.input_size[1]), dtype=np.float32)
                target_mask_occluded_minibatch = np.zeros((minibatch_size, 1, params.input_size[0], params.input_size[1]), dtype=np.float32)
                target_mask_eroded_minibatch = np.zeros((minibatch_size, 1, params.input_size[0], params.input_size[1]), dtype=np.float32)

                for i in range(0, minibatch_size):
                    paths = test_image_paths[image_index + i]

                    input_orig_img, target_uv_img, target_mask_img = get_images(paths, params.input_size)

                    input_aug_img, target_uv_img, target_mask_img, \
                    target_mask_occluded_img, target_mask_eroded_img = augment_images(input_orig_img, target_uv_img, target_mask_img,
                                                                                      params.test_augment, epoch_number.value, occlusion_textures)

                    input_image_data = convert_image_to_data(input_aug_img, DataTypes.Input)
                    target_uv_data = convert_image_to_data(target_uv_img, DataTypes.Uv)
                    target_mask_data = convert_image_to_data(target_mask_img, DataTypes.Mask)
                    target_mask_occluded_data = convert_image_to_data(target_mask_occluded_img, DataTypes.Mask)
                    target_mask_eroded_data = convert_image_to_data(target_mask_eroded_img, DataTypes.Mask)

                    input_image_minibatch[i] = input_image_data
                    target_uv_minibatch[i] = target_uv_data
                    target_mask_minibatch[i] = target_mask_data
                    target_mask_occluded_minibatch[i] = target_mask_occluded_data
                    target_mask_eroded_minibatch[i] = target_mask_eroded_data

                image_index += minibatch_size

                while True:
                    if should_stop.is_set() or interrupted:
                        break
                    if should_send.wait(0.1):
                        should_send.clear()
                        minibatch_pipe.send((input_image_minibatch, target_uv_minibatch, target_mask_minibatch, target_mask_occluded_minibatch, target_mask_eroded_minibatch))
                        break

            epoch_finished.set()
    except (KeyboardInterrupt, InterruptedError):
        pass
    except:
        print("\nData read process failed!\n")
        traceback.print_exc()

    minibatch_pipe.close()


def plot_results_process_func(params: Parameters,
                              test_results: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
                              real_results: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
                              epoch_number: int,
                              font_path: str,
                              plot_path: str,
                              face_texture_path: str,
                              html_template_path: str,
                              html_result_path: str,
                              run_id: str,
                              git_commit_name: str,
                              elapsed_time: str,
                              param_count: int,
                              samples_seen: int,
                              samples_per_s: float,
                              train_losses: List[float],
                              test_losses: List[float],
                              eval_metrics: List[float]) -> None:
    """Do results plotting in another process."""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        plot_all_results(test_results, real_results, epoch_number, git_commit_name, font_path, plot_path, face_texture_path)
        html_template = create_html_template(html_template_path)
        write_html(html_template, html_result_path, params, run_id, git_commit_name, epoch_number, elapsed_time,
                   param_count, samples_seen, samples_per_s, train_losses, test_losses, eval_metrics)
    except (KeyboardInterrupt, InterruptedError):
        pass
    except:
        print("\nPlot results process failed!\n")
        traceback.print_exc()


def do_training(gpu_device_id: int = None, params_file_path: str = None) -> int:
    """Main training loop"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if params_file_path is not None:
        params = load_parameters_from_file(params_file_path)
    else:
        params = setup_parameters()

    model = None
    output_path = None
    model_path = None
    read_data_process = None
    read_data_process_started = False
    plot_results_process = None
    plot_results_process_started = False
    minibatch_pipe, minibatch_pipe_child = mp.Pipe(duplex=False)
    read_data_process_should_stop = mp.Event()
    read_data_process_should_proceed = mp.Event()
    read_data_process_should_send = mp.Event()
    read_data_process_epoch_finished = mp.Event()
    epoch_number = mp.Value("L", 0)
    exit_code = 0

    try:
        git_commit_name = get_git_commit_tag_name()

        if git_commit_name is not None:
            run_id = "{0}-{1:%Y-%m-%d-%H-%M-%S}".format(git_commit_name, datetime.datetime.now())
        else:
            git_commit_name = get_git_commit_short_hash()
            run_id = "{0:%Y-%m-%d-%H-%M-%S}-{1}".format(datetime.datetime.now(), git_commit_name)

        abs_script_path = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(params.output_directory, "uvnet-{0}".format(run_id))
        model_path = os.path.join(output_path, "model")
        font_path = os.path.join(abs_script_path, "data/fonts/dejavu-sans-regular.ttf")
        eval_test_images_path = os.path.join(abs_script_path, "data/eval_test_images")
        eval_real_images_path = os.path.join(abs_script_path, "data/eval_real_images1")
        plot_path = os.path.join(output_path, "plot")
        html_template_path = os.path.join(abs_script_path, "results.html")
        html_result_path = os.path.join(output_path, "results.html")
        occlusion_textures_path = os.path.join(params.textures_directory, "environments")
        face_texture_path = os.path.join(params.textures_directory, "faces/1.png")

        # create directories
        for path in [output_path, model_path, plot_path]:
            os.makedirs(path)

        Logger(os.path.join(output_path, "uvnet.log"), "w")

        print("Starting training run {0}\n".format(run_id))

        if params_file_path is not None:
            print("Training run parameters (loaded from {0}):\n\n{1}\n".format(params_file_path, params.to_string()))
        else:
            print("Training run parameters (default):\n\n{0}\n".format(params.to_string()))

        print_code_as_html(os.path.join(abs_script_path, "uvnet.py"), os.path.join(output_path, "source.html"))
        print_code_as_html(os.path.join(abs_script_path, "uvnet_utils.py"), os.path.join(output_path, "source_utils.html"))

        train_image_paths = []
        test_image_paths = []
        real_image_aligned_paths = []
        real_image_paths = []
        train_losses = []
        test_losses = []
        eval_metrics = []

        for directory in params.train_image_directories:
            train_image_paths.extend(read_file_paths_as_triplets(directory, params.train_image_filter))

        for directory in params.test_image_directories:
            test_image_paths.extend(read_file_paths_as_triplets(directory, params.train_image_filter))

        for directory in params.real_image_aligned_directories:
            real_image_aligned_paths.extend(read_file_paths(directory))

        for directory in params.real_image_directories:
            real_image_paths.extend(read_file_paths(directory))

        eval_test_image_paths = read_file_paths_as_triplets(eval_test_images_path, ".exr")
        eval_real_image_paths = read_file_paths(eval_real_images_path)
        occlusion_texture_paths = read_file_paths(occlusion_textures_path, ".hdr")
        occlusion_textures = []

        for occlusion_texture_path in occlusion_texture_paths:
            occlusion_textures.append(normalize_image(read_image_from_file(occlusion_texture_path)))

        train_image_count = len(train_image_paths)
        test_image_count = len(test_image_paths)
        real_image_aligned_count = len(real_image_aligned_paths)
        real_image_count = len(real_image_paths)
        train_minibatch_count = math.ceil(params.train_epoch_size / params.train_minibatch_size)
        test_minibatch_count = math.ceil(params.test_epoch_size / params.test_minibatch_size)

        print("\nTrain images: {0}".format(train_image_count))
        print("Test images: {0}".format(test_image_count))
        print("Real aligned images: {0}".format(real_image_aligned_count))
        print("Real images: {0}".format(real_image_count))

        assert params.train_epoch_size <= train_image_count
        assert params.train_minibatch_size <= params.train_epoch_size
        assert params.test_epoch_size <= test_image_count
        assert params.test_minibatch_size <= params.test_epoch_size

        cntk.set_excluded_devices([cntk.cpu()])

        if gpu_device_id is not None:
            print("\nTrying to select GPU with device id {0}".format(gpu_device_id))
            cntk.try_set_default_device(cntk.gpu(gpu_device_id), acquire_device_lock=False)
        else:
            print("\nTarget GPU will be selected automatically")

        gpu_device = cntk.use_default_device()
        gpu_desc = cntk.get_gpu_properties(gpu_device)
        print("Selected GPU: {0}:{1}".format(gpu_desc.device_id, gpu_desc.name))

        print("\nInitializing CNTK model and variables")

        model, loss_function, eval_function, learner, learning_rate_schedule, momentum_schedule, \
        input_image, target_uv, target_mask, target_mask_occluded, target_mask_eroded = setup_cntk(params)

        progress_printer = cntk.logging.ProgressPrinter(num_epochs=params.train_epoch_count, metric_is_pct=False, freq=10, test_freq=10)
        trainer = cntk.Trainer(model, (loss_function, loss_function), [learner], [progress_printer])
        trainer_for_eval = cntk.Trainer(model, (loss_function, eval_function), [learner])

        param_count = calculate_parameter_count(model)
        print("Total model parameters: {0}\n".format(param_count))

        if params.model_load_from_file:
            print("\nLoading model parameters from {0}".format(params.model_load_file_name))
            model.restore(params.model_load_file_name)

        read_data_process = mp.Process(target=read_data_process_func, name="ReadData", args=(params, epoch_number, minibatch_pipe_child, read_data_process_should_stop,
                                                                                             read_data_process_should_proceed, read_data_process_should_send,
                                                                                             read_data_process_epoch_finished, train_image_paths, test_image_paths,
                                                                                             occlusion_texture_paths))
        read_data_process.start()
        read_data_process_started = True

        samples_seen = 0
        fixed_random = random.Random(1234)
        train_start_time = time.time()

        for epoch in range(1, params.train_epoch_count + 1):
            print("\nStarting epoch {0} | Learning rate: {1} | Momentum: {2}\n".format(epoch, learning_rate_schedule[samples_seen], momentum_schedule[samples_seen]))
            epoch_number.value = epoch

            train_loss_sum = 0.0
            minibatch_read_time_sum = 0.0
            minibatch_train_time_sum = 0.0
            read_data_process_should_proceed.set()

            while not read_data_process_epoch_finished.is_set() and not interrupted:
                start_time = time.time()
                read_data_process_should_send.set()
                (input_image_minibatch, target_uv_minibatch, target_mask_minibatch, target_mask_occluded_minibatch, target_mask_eroded_minibatch) = minibatch_pipe.recv()
                minibatch_read_time_sum += time.time() - start_time
                start_time = time.time()
                trainer.train_minibatch({input_image: input_image_minibatch, target_uv: target_uv_minibatch,
                                         target_mask: target_mask_minibatch, target_mask_occluded: target_mask_occluded_minibatch,
                                         target_mask_eroded: target_mask_eroded_minibatch})
                minibatch_train_time_sum += time.time() - start_time
                train_loss_sum += trainer.previous_minibatch_loss_average

            read_data_process_epoch_finished.clear()

            if not interrupted:
                trainer.summarize_training_progress()
                print("")

            test_loss_sum = 0.0
            eval_metric_sum = 0.0
            read_data_process_should_proceed.set()

            while not read_data_process_epoch_finished.is_set() and not interrupted:
                read_data_process_should_send.set()
                (input_image_minibatch, target_uv_minibatch, target_mask_minibatch, target_mask_occluded_minibatch, target_mask_eroded_minibatch) = minibatch_pipe.recv()
                test_loss_sum += trainer.test_minibatch({input_image: input_image_minibatch, target_uv: target_uv_minibatch,
                                                         target_mask: target_mask_minibatch, target_mask_occluded: target_mask_occluded_minibatch,
                                                         target_mask_eroded: target_mask_eroded_minibatch})
                eval_metric_sum += trainer_for_eval.test_minibatch({input_image: input_image_minibatch, target_uv: target_uv_minibatch,
                                                                    target_mask: target_mask_minibatch})

            read_data_process_epoch_finished.clear()

            if not interrupted:
                trainer.summarize_test_progress()
                print("")

                elapsed_seconds = time.time() - train_start_time
                m, s = divmod(elapsed_seconds, 60)
                h, m = divmod(m, 60)
                elapsed_time = "{0:02.0f}:{1:02.0f}:{2:02.0f}".format(h, m, s)

                train_losses.append(train_loss_sum / train_minibatch_count)
                test_losses.append(test_loss_sum / test_minibatch_count)
                eval_metrics.append(eval_metric_sum / test_minibatch_count)

                print("Average minibatch read time: {0:.2f} ms\nAverage minibatch train time: {1:.2f} ms".format((minibatch_read_time_sum / train_minibatch_count) * 1000.0,
                                                                                                                 (minibatch_train_time_sum / train_minibatch_count) * 1000.0))

                samples_seen = trainer.total_number_of_samples_seen
                samples_per_s = samples_seen / elapsed_seconds

                print("\nEvaluating the model...")
                test_results, real_results = evaluate_model_multiple(model, params,
                                                                     eval_test_image_paths + [fixed_random.choice(test_image_paths)],
                                                                     eval_real_image_paths + [fixed_random.choice(real_image_aligned_paths)],
                                                                     [fixed_random.choice(real_image_paths)],
                                                                     occlusion_textures, epoch)

                if plot_results_process_started:
                    if plot_results_process.is_alive():
                        print("Waiting for the plot results process to end")
                    plot_results_process.join()

                plot_results_process = mp.Process(target=plot_results_process_func, name="PlotResults",
                                                  args=(params, test_results, real_results, epoch, font_path, plot_path, face_texture_path, html_template_path, html_result_path,
                                                        run_id, git_commit_name, elapsed_time, param_count, samples_seen, samples_per_s, train_losses, test_losses, eval_metrics))
                plot_results_process.start()
                plot_results_process_started = True

                if params.model_save_intermediate and epoch % params.model_save_interval == 0:
                    print("\nSaving model parameters")
                    model.save(os.path.join(model_path, "uvnet_{0}.model".format(epoch)))

                print("\nTime elapsed: {0} | Total samples seen: {1}".format(elapsed_time, samples_seen))

                if elapsed_seconds >= params.max_training_time:
                    print("\nMax training time reached!")
                    break
            else:
                break

        print("\nTraining finished!\n")
    except (KeyboardInterrupt, InterruptedError):
        pass
    except:
        print("\nTraining failed!\n")
        traceback.print_exc()
        exit_code = 42

    if params.model_save_final and model is not None:
        print("\nSaving model parameters")
        model.save(os.path.join(output_path, "uvnet.model"))
        shutil.rmtree(model_path)  # delete the intermediate model files

    read_data_process_should_stop.set()

    if plot_results_process_started:
        if plot_results_process.is_alive():
            print("Waiting for the plot results process to end")
        plot_results_process.join()

    if read_data_process_started:
        if read_data_process.is_alive():
            print("Waiting for the read data process to end")
        read_data_process.join()

    minibatch_pipe.close()
    print("\nDone")

    return exit_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, help="The CUDA GPU device id")
    parser.add_argument("--params", type=str, help="Path to serialized run parameters file")
    args = parser.parse_args()

    result = do_training(args.gpu, args.params)
    sys.exit(result)
