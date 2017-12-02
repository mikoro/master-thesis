# coding=utf-8

"""Miscellaneous testing function."""

from uvnet import *

abs_script_path = os.path.dirname(os.path.abspath(__file__))
occlusion_texture_paths = read_file_paths("D:\\Thesis\\textures\\environments", ".hdr")
head_expressions_data_path = os.path.join(abs_script_path, "../blender/head/models/head_expressions.dat")
font_path = os.path.join(abs_script_path, "data/fonts/dejavu-sans-regular.ttf")
model_file_path = os.path.join(abs_script_path, "uvnet.model")
face_texture_path = "D:\\Thesis\\textures\\faces\\1.png"
face_texture_img = normalize_image(read_image_from_file(face_texture_path), fix_gamma=True)

train_image_paths = read_file_paths_as_triplets("D:\\Thesis\\renders", ".exr")
# real_image_aligned_paths = read_file_paths("C:\\Thesis\\misc\\real_aligned_test")
real_image_aligned_paths = []
# real_image_paths = read_file_paths("C:\\Thesis\\misc\\real_test")
real_image_paths = []
eval_test_image_paths = read_file_paths_as_triplets(os.path.join(abs_script_path, "data/eval_test_images"), ".exr")
eval_real_image_paths = read_file_paths(os.path.join(abs_script_path, "data/eval_real_images1"))

occlusion_textures = []
collage_images = []


def read_occlusion_textures():
    print("Reading occlusion textures")

    for texture_path in occlusion_texture_paths:
        occlusion_textures.append(normalize_image(read_image_from_file(texture_path)))

    print("Finished reading occlusion textures")


def create_collage():
    read_occlusion_textures()

    augment = AugmentParameters()
    augment.enabled = True
    augment.occlusion = True
    augment.shuffle = False
    augment.rotate = False
    augment.exposure = False
    augment.gamma = False
    augment.noise = False
    augment.clip = True
    augment.quantize = True
    augment.occlusion_prob = 1.0
    augment.shuffle_prob = 1.0

    fixed_random = random.Random(9)
    random_paths = fixed_random.sample(train_image_paths, 35)
    #random_paths = train_image_paths[:35]

    for random_path in random_paths:
        input_image_img, target_uv_img, target_mask_img = get_images(random_path, (128, 128))

        input_image_img2, target_uv_img2, target_mask_img2, \
        target_mask_occluded_img2, target_mask_eroded_img2 = augment_images(input_image_img, target_uv_img, target_mask_img, augment, 0, occlusion_textures)

        collage_images.append(input_image_img2)
        #collage_images.append(target_uv_img2)
        #collage_images.append(target_mask_img2)
        #collage_images.append(target_mask_occluded_img2)

    create_image_collage(collage_images, 5, "C:\\Users\\mikor\\Desktop\\head_augment_occlusion_collage.png")


def augment_image():
    read_occlusion_textures()

    augment = AugmentParameters()
    augment.enabled = True
    augment.occlusion = True
    augment.shuffle = True
    augment.rotate = True
    augment.exposure = True
    augment.gamma = True
    augment.noise = True
    augment.clip = True
    augment.quantize = True
    augment.shuffle_prob = 1.0
    augment.occlusion_prob = 1.0
    augment.rotate_prob = 1.0

    input_image_img, target_uv_img, target_mask_img = get_images(eval_test_image_paths[0], (128, 128))

    for i in range(20):
        input_image_img2, target_uv_img2, target_mask_img2, \
        mask_occluded_img2, target_mask_eroded_img2 = augment_images(input_image_img, target_uv_img, target_mask_img, augment, 0, occlusion_textures)

        save_image_to_file(prepare_image_for_viewing(input_image_img2), "C:\\Users\\mikor\\Desktop\\head_augment_all{0}.png".format(i))

    #save_image_to_file(prepare_image_for_viewing(input_image_img2), "C:\\Users\\mikor\\Desktop\\train_sample2.png")
    #save_image_to_file(prepare_image_for_viewing(target_uv_img2), "C:\\Users\\mikor\\Desktop\\train_sample3.png")
    #save_image_to_file(prepare_image_for_viewing(target_mask_img2), "C:\\Users\\mikor\\Desktop\\train_sample4.png")
    #save_image_to_file(prepare_image_for_viewing(target_mask_eroded_img2), "C:\\Users\\mikor\\Desktop\\train_sample6.png")
    #save_image_to_file(prepare_image_for_viewing(mask_occluded_img2), "C:\\Users\\mikor\\Desktop\\train_sample7.png")


def modify_expressions_data():
    head_expressions_data = pickle.load(open(head_expressions_data_path, "rb"))
    d1 = np.zeros((1, 8104, 3), dtype=np.float32)
    head_expressions_data = np.concatenate((head_expressions_data, d1))
    head_expressions_data = np.concatenate((head_expressions_data, d1))
    pickle.dump(head_expressions_data, open(head_expressions_data_path, "wb"))


def evaluate_and_plot_model():
    read_occlusion_textures()

    params = setup_parameters()
    # params.eval_augment.occlusion_prob = 1.0

    model, loss_function, eval_function, learner, learning_rate_schedule, momentum_schedule, \
    input_image, target_uv, target_mask, target_mask_occluded, target_mask_eroded = setup_cntk(params)

    model.restore(model_file_path)

    for i in range(1, 21):
        test_results, real_results = evaluate_model_multiple(model, params,
                                                             eval_test_image_paths + [random.choice(train_image_paths)],
                                                             eval_real_image_paths + [random.choice(real_image_aligned_paths)],
                                                             [random.choice(real_image_paths)],
                                                             occlusion_textures, i)

        plot_all_results(test_results, real_results, i, "net-test", font_path, "C:\\Users\\Mikko\\Desktop\\plots", face_texture_path)


def evaluate_and_create_collage():
    read_occlusion_textures()

    params = setup_parameters()
    params.use_random_eval_augment = False
    params.eval_augment.noise = False
    params.eval_augment.occlusion_prob = 0.5

    model, loss_function, eval_function, learner, learning_rate_schedule, momentum_schedule, \
    input_image, target_uv, target_mask, target_mask_occluded, target_mask_eroded = setup_cntk(params)

    model.restore(model_file_path)

    for i in range(20):
        collage_images.clear()
        test_results, real_results = evaluate_model_multiple(model, params,
                                                             random.sample(train_image_paths, 10),
                                                             eval_real_image_paths,
                                                             [],
                                                             occlusion_textures, 1)
        for test_result in test_results:
            collage_images.append(test_result[0])
            collage_images.append(test_result[1])
            collage_images.append(test_result[3])
            collage_images.append(test_result[7])
            collage_images.append(test_result[4])
            collage_images.append(test_result[8])
            collage_images.append(test_result[2])
            collage_images.append(test_result[6] * test_result[7])

        create_image_collage(collage_images, 8, "C:\\Users\\mikor\\Desktop\\test_results_collage{0}.png".format(i))

    # collage_images.clear()
    #
    # for real_result in real_results:
    #     collage_images.append(real_result[0])
    #     collage_images.append(real_result[2])
    #     collage_images.append(real_result[3])
    #     collage_images.append(real_result[1] * real_result[2])
    #
    #     texture_proj_img = get_texture_projection(face_texture_img, real_result[1], real_result[2])
    #     texture_proj_inv_img = get_texture_projection_inv(real_result[0], real_result[1], real_result[2])
    #     texture_proj_input_img = get_texture_projection_input(face_texture_img, real_result[0], real_result[1], real_result[2])
    #     grid_lines_proj_img = get_grid_lines_projection(real_result[0], real_result[1], real_result[2])
    #     grid_points_proj_img = get_grid_points_projection(real_result[0], real_result[1], real_result[2])
    #
    #     collage_images.append(texture_proj_input_img)
    #     collage_images.append(grid_lines_proj_img)
    #     collage_images.append(grid_points_proj_img)
    #     collage_images.append(texture_proj_inv_img)
    #
    # create_image_collage(collage_images, 8, "C:\\Users\\mikor\\Desktop\\real_results_collage.png")


def evaluate_model():
    read_occlusion_textures()

    params = setup_parameters()

    model, loss_function, eval_function, learner, learning_rate_schedule, momentum_schedule, \
    input_image, target_uv, target_mask, target_mask_occluded, target_mask_eroded = setup_cntk(params)

    model.restore(model_file_path)

    augment = AugmentParameters()
    augment.enabled = True
    augment.occlusion = True
    augment.shuffle = False
    augment.rotate = False
    augment.exposure = False
    augment.gamma = False
    augment.noise = False
    augment.clip = False
    augment.quantize = False
    augment.occlusion_prob = 1.0

    input_image_img, target_uv_img, target_mask_img = get_images(eval_test_image_paths[0], (128, 128))

    input_image_aug_img, target_uv_img, target_mask_img, \
    target_mask_occl_img, target_mask_erod_img = augment_images(input_image_img, target_uv_img, target_mask_img, augment, 0, occlusion_textures)

    result_uv_img, result_mask_img, result_mask_occluded_img = evaluate_model_single(model, input_image_aug_img)
    result_uv_masked_img = result_uv_img * target_mask_img

    save_image_to_file(prepare_image_for_viewing(input_image_img), "train_sample_input.png")
    save_image_to_file(prepare_image_for_viewing(input_image_aug_img), "train_sample_input_aug.png")
    save_image_to_file(prepare_image_for_viewing(target_uv_img), "train_sample_target_uv.png")
    save_image_to_file(prepare_image_for_viewing(target_mask_img), "train_sample_target_mask.png")
    save_image_to_file(prepare_image_for_viewing(target_mask_occl_img), "train_sample_target_mask_occl.png")
    save_image_to_file(prepare_image_for_viewing(target_mask_erod_img), "train_sample_target_mask_erod.png")

    save_image_to_file(prepare_image_for_viewing(result_uv_img), "train_sample_result_uv.png")
    save_image_to_file(prepare_image_for_viewing(result_uv_masked_img), "train_sample_result_uv_masked.png")
    save_image_to_file(prepare_image_for_viewing(result_mask_img), "train_sample_result_mask.png")
    save_image_to_file(prepare_image_for_viewing(result_mask_occluded_img), "train_sample_result_mask_occl.png")

    _, _, _, _, u_grad_mag, v_grad_mag = get_image_gradients(target_uv_img, 2.0)

    save_image_to_file(prepare_image_for_viewing(u_grad_mag), "train_sample_target_u_grad.png")
    save_image_to_file(prepare_image_for_viewing(u_grad_mag * target_mask_erod_img), "train_sample_target_u_grad_masked.png")
    save_image_to_file(prepare_image_for_viewing(v_grad_mag), "train_sample_target_v_grad.png")
    save_image_to_file(prepare_image_for_viewing(v_grad_mag * target_mask_erod_img), "train_sample_target_v_grad_masked.png")

    _, _, _, _, u_grad_mag2, v_grad_mag2 = get_image_gradients(result_uv_img, 2.0)

    save_image_to_file(prepare_image_for_viewing(u_grad_mag2), "train_sample_result_u_grad.png")
    save_image_to_file(prepare_image_for_viewing(u_grad_mag2 * target_mask_erod_img), "train_sample_result_u_grad_masked.png")
    save_image_to_file(prepare_image_for_viewing(v_grad_mag2), "train_sample_result_v_grad.png")
    save_image_to_file(prepare_image_for_viewing(v_grad_mag2 * target_mask_erod_img), "train_sample_result_v_grad_masked.png")

    uv_diff_img = get_images_diff(target_uv_img, result_uv_masked_img)
    mask_diff_img = get_images_diff(target_mask_img, result_mask_img)
    mask_occluded_diff_img = get_images_diff(target_mask_occl_img, result_mask_occluded_img)
    grad_u_mag_diff_img = get_images_diff(u_grad_mag * target_mask_erod_img, u_grad_mag2 * target_mask_erod_img, 4.0)
    grad_v_mag_diff_img = get_images_diff(v_grad_mag * target_mask_erod_img, v_grad_mag2 * target_mask_erod_img, 4.0)

    save_image_to_file(prepare_image_for_viewing(uv_diff_img), "train_sample_uv_diff.png")
    save_image_to_file(prepare_image_for_viewing(mask_diff_img), "train_sample_mask_diff.png")
    save_image_to_file(prepare_image_for_viewing(mask_occluded_diff_img), "train_sample_mask_occl_diff.png")
    save_image_to_file(prepare_image_for_viewing(grad_u_mag_diff_img), "train_sample_grad_u_diff.png")
    save_image_to_file(prepare_image_for_viewing(grad_v_mag_diff_img), "train_sample_grad_v_diff.png")


def test_projections():
    params = setup_parameters()

    model, loss_function, eval_function, learner, learning_rate_schedule, momentum_schedule, \
    input_image, target_uv, target_mask, target_mask_occluded, target_mask_eroded = setup_cntk(params)

    model.restore(model_file_path)

    input_img = read_image_from_file(eval_real_image_paths[0])
    #input_img = input_img[20:198, :, :]
    input_img = normalize_image(input_img, params.input_size, fix_gamma=True)

    result_uv_img, result_mask_img, result_mask_occluded_img = evaluate_model_single(model, input_img)

    texture_proj_img = get_texture_projection(face_texture_img, result_uv_img, result_mask_img)
    texture_proj_inv_img = get_texture_projection_inv(input_img, result_uv_img, result_mask_img)
    texture_proj_input_img = get_texture_projection_input(face_texture_img, input_img, result_uv_img, result_mask_img)
    grid_lines_proj_img = get_grid_lines_projection(input_img, result_uv_img, result_mask_img)
    grid_points_proj_img = get_grid_points_projection(input_img, result_uv_img, result_mask_img)

    # show_image(texture_proj_img)
    # show_image(texture_proj_input_img)
    # show_image(grid_lines_proj_img)
    # show_image(grid_points_proj_img)

    save_image_to_file(prepare_image_for_viewing(input_img), "input_img.png")
    save_image_to_file(prepare_image_for_viewing(result_mask_img), "result_mask_img.png")
    save_image_to_file(prepare_image_for_viewing(result_uv_img * result_mask_img), "result_uv_img.png")
    save_image_to_file(prepare_image_for_viewing(texture_proj_img), "texture_proj_img.png")
    save_image_to_file(prepare_image_for_viewing(texture_proj_input_img), "texture_proj_input_img.png")
    save_image_to_file(prepare_image_for_viewing(grid_lines_proj_img), "grid_lines_proj_img.png")
    #save_image_to_file(prepare_image_for_viewing(grid_points_proj_img), "grid_points_proj_img.png")


def test_get_images():
    input_image_img, target_uv_img, target_mask_img = get_images(train_image_paths[0], (128, 128))
    save_image_to_file(prepare_image_for_viewing(input_image_img), "input2.png")
    save_image_to_file(prepare_image_for_viewing(target_uv_img), "uv2.png")
    save_image_to_file(prepare_image_for_viewing(target_mask_img), "mask2.png")
    show_image(input_image_img)
    show_image(target_uv_img)
    show_image(target_mask_img)


def test_image_gradients():
    read_occlusion_textures()

    input_image_img, target_uv_img, target_mask_img = get_images(train_image_paths[0], (128, 128))

    augment = AugmentParameters()
    augment.enabled = False
    augment.occlusion = False
    augment.shuffle = False
    augment.rotate = False
    augment.exposure = False
    augment.gamma = False
    augment.noise = False
    augment.clip = False
    augment.quantize = False
    augment.mask_erosion_amount = 3

    input_image_aug_img, target_uv_img, target_mask_img, \
    target_mask_occl_img, target_mask_erod_img = augment_images(input_image_img, target_uv_img, target_mask_img, augment, 0, occlusion_textures)

    ux_grad, uy_grad, vx_grad, vy_grad, u_grad_mag, v_grad_mag = get_image_gradients(target_uv_img, 2.0)

    # show_image(ux_grad)
    # show_image(uy_grad)
    # show_image(vx_grad)
    # show_image(vy_grad)
    show_image(u_grad_mag * target_mask_erod_img)
    show_image(v_grad_mag * target_mask_erod_img)


def test_schedules():
    learning_rate_schedule = cntk.learning_rate_schedule([(5, 1), (5, 2), (5, 3), (5, 4)], unit=cntk.UnitType.minibatch, epoch_size=10)
    momentum_schedule = cntk.momentum_schedule([(5, 1), (5, 2), (5, 3), (5, 4)], epoch_size=10)
    my_schedule = [(5, 1), (5, 2), (5, 3), (5, 4)]

    epoch_count = 30
    epoch_size = 10
    minibatch_size = 2
    samples_seen = 0

    for epoch in range(epoch_count):
        index = 0

        while index < epoch_size:
            print("{0} | {1} | {2} | {3} | {4}".format(epoch, samples_seen, learning_rate_schedule[samples_seen], momentum_schedule[samples_seen], get_sched_value(my_schedule, epoch + 1)))
            index += minibatch_size
            samples_seen += minibatch_size


# create_collage()
# augment_image()
# evaluate_and_plot_model()
# evaluate_and_create_collage()
evaluate_model()
# test_image_gradients()
# test_get_images()
# test_schedules()
# test_projections()
