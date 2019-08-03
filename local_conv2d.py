def local_conv2d(inputs, kernel, kernel_size, strides, output_shape, data_format=None):
    data_format = normalize_data_format(data_format)

    stride_row, stride_col = strides
    output_row, output_col = output_shape
    kernel_shape = int_shape(kernel)
    _, dim, filters = kernel_shape
    kernel_row, kernel_col = kernel_size[0], kernel_size[1]

    # result: (b, output_row * output_col, kernel_row * kernel_col * input_filter)
    image_patches = tf.extract_image_patches(inputs,
                                             [1, kernel_row, kernel_col, 1],
                                             [1, stride_row, stride_col, 1],
                                             [1, 1, 1, 1],
                                             padding='VALID')

    image_patches = tf.reshape(image_patches, [-1, output_row * output_col, dim])
    image_patches = tf.transpose(image_patches, [1, 0, 2])
    output = K.batch_dot(image_patches, kernel)
    output = reshape(output,
                     (output_row, output_col, -1, filters))

    if data_format == 'channels_first':
        output = permute_dimensions(output, (2, 3, 0, 1))
    else:
        output = permute_dimensions(output, (2, 0, 1, 3))
    return output
