def cntk_l1_loss(a, b) -> cntk.Function:
  return cntk.ops.reduce_sum(cntk.ops.abs(a - b))

def setup_cntk(params: Parameters) -> Tuple[cntk.Function, cntk.Function, cntk.Function]:
  input_image = cntk.input_variable((3, params.input_size[0], params.input_size[1]))
  target_uv = cntk.input_variable((2, params.input_size[0], params.input_size[1]))
  target_mask = cntk.input_variable((1, params.input_size[0], params.input_size[1]))
  target_mask_occluded = cntk.input_variable((1, params.input_size[0], params.input_size[1]))
  target_mask_eroded = cntk.input_variable((1, params.input_size[0], params.input_size[1]))

  model = create_model(input_image, params)

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

  return model, loss_function, eval_function