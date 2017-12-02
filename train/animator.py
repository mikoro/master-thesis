# coding=utf-8

"""Create animation using a video and a model file."""

from uvnet import *
from uvnet_utils import *

vc = cv2.VideoCapture("C:\\Users\\mikor\\Desktop\\videos\\input videos\\input_video3.mp4")
params = Parameters()

model, loss_function, eval_function, learner, learning_rate_schedule, momentum_schedule, \
input_image, target_uv, target_mask, target_mask_occluded, target_mask_eroded = setup_cntk(params)

abs_script_path = os.path.dirname(os.path.abspath(__file__))

model.restore(os.path.join(abs_script_path, "uvnet.model"))

face_texture_path = os.path.join(abs_script_path, "D:\\Thesis\\textures\\faces\\1.png")
font_path = os.path.join(abs_script_path, "data/fonts/dejavu-sans-regular.ttf")
face_texture_img = normalize_image(read_image_from_file(face_texture_path), fix_gamma=True)

start_frame = 13501
stop_frame = 18000
frame_counter = 0
texture_proj_inv_images = np.zeros((30, 128, 128, 3), dtype=np.float32)

while True:
    ret, input_img = vc.read()
    frame_counter += 1

    if frame_counter < start_frame:
        continue

    if frame_counter > stop_frame:
        break

    if not ret:
        break

    input_img = (cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB) / 255.0) ** 2.2
    # input_img = np.rot90(input_img, 1)
    # input_img = input_img[420:-420, :, :]
    input_img = cv2.resize(input_img, (128, 128), interpolation=cv2.INTER_AREA)

    result_uv_img, result_mask_img, result_mask_occluded_img = evaluate_model_single(model, input_img)

    plot_real_result2(input_img,
                      result_uv_img,
                      result_mask_img,
                      result_mask_occluded_img,
                      texture_proj_inv_images,
                      face_texture_img,
                      frame_counter,
                      font_path,
                      "C:\\Users\\mikor\\Desktop\\videos\\frames\\6")

    print("Plotted frame {0}".format(frame_counter))
