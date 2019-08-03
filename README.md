# keras-local_conv2d
Keras performs LocallyConnected2D convolution with unshared convolution parameters, improving local_conv2d in Keras tensorflow_backend
使用keras实现本地连接2D卷积

## 改进之处
使用`tf.extract_image_patches`代替原有的双层循环，速度上原来要快。

## 代码解析
* 将原图处理成图像块，相当于im2col函数的功能，产生形状为`(batch, output_row * output_col, kernel_row * kernel_col * input_filter)`
* kernel形状：`（output_row * output_col, kernel_size[0] * kernel_size[1] * input_filter, output_filters)`
* 将卷积运算转换为矩阵乘法，使用`K.batch_dot`函数
* 将输出形状和维度进行转换，输出形状为`(batch, output_row, output_col, output_filters)`
## 使用
* `from local import LocallyConnected2D`
*`x = LocallyConnected2D(32, (3, 3), activation='relu')(x)`
