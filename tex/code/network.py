def create_model(input_image: cntk.Variable, params: Parameters) -> cntk.Function:
  cnf = [int(round(params.model.initial_features * pow(params.model.feature_multiplier, i))) for i in range(0, 7)]
  ucnf = [int(round(params.model.up_factor * i)) for i in cnf]
  fs = params.model.filter_size

  with cntk.default_options(init=cntk.glorot_uniform(), activation=cntk.relu, pad=True, bias=True):
    p1, p2, p3, p4, p5, p6 = None, None, None, None, None, None

    l = cntk.layers.Convolution(filter_shape=fs, strides=(1, 1), num_filters=cnf[0])(input_image)
    l = cntk.layers.Convolution(filter_shape=fs, strides=(1, 1), num_filters=cnf[0])(l)
    p1 = l = cntk.layers.MaxPooling(filter_shape=(2, 2), strides=(2, 2))(l)

    if params.model.levels >= 2:
      l = cntk.layers.Convolution(filter_shape=fs, strides=(1, 1), num_filters=cnf[1])(l)
      l = cntk.layers.Convolution(filter_shape=fs, strides=(1, 1), num_filters=cnf[1])(l)
      p2 = l = cntk.layers.MaxPooling(filter_shape=(2, 2), strides=(2, 2))(l)

    if params.model.levels >= 3:
      l = cntk.layers.Convolution(filter_shape=fs, strides=(1, 1), num_filters=cnf[2])(l)
      l = cntk.layers.Convolution(filter_shape=fs, strides=(1, 1), num_filters=cnf[2])(l)
      p3 = l = cntk.layers.MaxPooling(filter_shape=(2, 2), strides=(2, 2))(l)

    if params.model.levels >= 4:
      l = cntk.layers.Convolution(filter_shape=fs, strides=(1, 1), num_filters=cnf[3])(l)
      l = cntk.layers.Convolution(filter_shape=fs, strides=(1, 1), num_filters=cnf[3])(l)
      p4 = l = cntk.layers.MaxPooling(filter_shape=(2, 2), strides=(2, 2))(l)

    if params.model.levels >= 5:
      l = cntk.layers.Convolution(filter_shape=fs, strides=(1, 1), num_filters=cnf[4])(l)
      l = cntk.layers.Convolution(filter_shape=fs, strides=(1, 1), num_filters=cnf[4])(l)
      p5 = l = cntk.layers.MaxPooling(filter_shape=(2, 2), strides=(2, 2))(l)

    if params.model.levels >= 6:
      l = cntk.layers.Convolution(filter_shape=(3, 3), strides=(1, 1), num_filters=cnf[5])(l)
      l = cntk.layers.Convolution(filter_shape=(3, 3), strides=(1, 1), num_filters=cnf[5])(l)
      p6 = l = cntk.layers.MaxPooling(filter_shape=(2, 2), strides=(2, 2))(l)

    if params.model.levels >= 7:
      l = cntk.layers.Convolution(filter_shape=(1, 1), strides=(1, 1), num_filters=cnf[6])(l)
      l = cntk.layers.Convolution(filter_shape=(1, 1), strides=(1, 1), num_filters=cnf[6])(l)
      l = cntk.layers.MaxPooling(filter_shape=(2, 2), strides=(2, 2))(l)

    if params.model.levels >= 7:
      l = cntk.layers.ConvolutionTranspose(filter_shape=(2, 2), strides=(2, 2),
          num_filters=ucnf[6], output_shape=(2, 2))(l)
      l = cntk.layers.Convolution(filter_shape=(1, 1), strides=(1, 1), num_filters=ucnf[6])(l)
      l = cntk.layers.Convolution(filter_shape=(1, 1), strides=(1, 1), num_filters=ucnf[6])(l)
      l = cntk.ops.splice(l, p6, axis=0)

    if params.model.levels >= 6:
      l = cntk.layers.ConvolutionTranspose(filter_shape=(3, 3), strides=(2, 2),
          num_filters=ucnf[5], output_shape=(4, 4))(l)
      l = cntk.layers.Convolution(filter_shape=(3, 3), strides=(1, 1), num_filters=ucnf[5])(l)
      l = cntk.layers.Convolution(filter_shape=(3, 3), strides=(1, 1), num_filters=ucnf[5])(l)
      l = cntk.ops.splice(l, p5, axis=0)

    if params.model.levels >= 5:
      l = cntk.layers.ConvolutionTranspose(filter_shape=fs, strides=(2, 2),
          num_filters=ucnf[4], output_shape=(8, 8))(l)
      l = cntk.layers.Convolution(filter_shape=fs, strides=(1, 1), num_filters=ucnf[4])(l)
      l = cntk.layers.Convolution(filter_shape=fs, strides=(1, 1), num_filters=ucnf[4])(l)
      l = cntk.ops.splice(l, p4, axis=0)

    if params.model.levels >= 4:
      l = cntk.layers.ConvolutionTranspose(filter_shape=fs, strides=(2, 2),
          num_filters=ucnf[3], output_shape=(16, 16))(l)
      l = cntk.layers.Convolution(filter_shape=fs, strides=(1, 1), num_filters=ucnf[3])(l)
      l = cntk.layers.Convolution(filter_shape=fs, strides=(1, 1), num_filters=ucnf[3])(l)
      l = cntk.ops.splice(l, p3, axis=0)

    if params.model.levels >= 3:
      l = cntk.layers.ConvolutionTranspose(filter_shape=fs, strides=(2, 2),
          num_filters=ucnf[2], output_shape=(32, 32))(l)
      l = cntk.layers.Convolution(filter_shape=fs, strides=(1, 1), num_filters=ucnf[2])(l)
      l = cntk.layers.Convolution(filter_shape=fs, strides=(1, 1), num_filters=ucnf[2])(l)
      l = cntk.ops.splice(l, p2, axis=0)

    if params.model.levels >= 2:
      l = cntk.layers.ConvolutionTranspose(filter_shape=fs, strides=(2, 2),
          num_filters=ucnf[1], output_shape=(64, 64))(l)
      l = cntk.layers.Convolution(filter_shape=fs, strides=(1, 1), num_filters=ucnf[1])(l)
      l = cntk.layers.Convolution(filter_shape=fs, strides=(1, 1), num_filters=ucnf[1])(l)
      l = cntk.ops.splice(l, p1, axis=0)

    l = cntk.layers.ConvolutionTranspose(filter_shape=fs, strides=(2, 2),
        num_filters=ucnf[0], output_shape=(128, 128))(l)
    l = cntk.layers.Convolution(filter_shape=fs, strides=(1, 1), num_filters=ucnf[0])(l)
    l = cntk.layers.Convolution(filter_shape=fs, strides=(1, 1), num_filters=ucnf[0])(l)
    l = cntk.ops.splice(l, input_image, axis=0)

    l = cntk.layers.Convolution(filter_shape=fs, strides=(1, 1), num_filters=ucnf[0])(l)
    l = cntk.layers.Convolution(filter_shape=fs, strides=(1, 1), num_filters=ucnf[0])(l)
    l = cntk.layers.Convolution(filter_shape=fs, strides=(1, 1), num_filters=4, activation=None)(l)

  return l