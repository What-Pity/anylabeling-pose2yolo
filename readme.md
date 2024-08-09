# 简介

将`labelme`生成的`JSON`文件转换为`YOLO`格式的`txt`文件。

# 使用方法

## 使用`labelme`标注

根据`Ultralytics YOLO-POSE`的[相关文档](https://docs.ultralytics.com/datasets/pose/)，`YOLO`支持的标注格式如下：

1. 每张图片对应一个文本文件：数据集中的每张图片都有一个与之对应的文本文件，该文件的名称与图片文件相同，扩展名为".txt"。
2. 每行描述一个对象：文本文件中的每一行对应图片中的一个对象实例。
3. 每行包含对象信息：每一行包含关于对象实例的以下信息：
   - 对象类别索引：一个整数，代表对象的类别（例如，0表示人，1表示汽车等）。
   - 对象中心坐标：对象中心的x和y坐标，归一化到0到1之间。
   - 对象宽度和高度：对象的宽度和高度，归一化到0到1之间。
   - 对象关键点坐标：对象的关键点坐标，归一化到0到1之间。

    ```txt
    <class-index> <x> <y> <width> <height> <px1> <py1> <p1-visibility> <px2> <py2> <p2-visibility> <pxn> <pyn> <p2-visibility>
    ```

    在这个格式中，`<class-index>` 是对象所属类别的索引，`<x> <y> <width> <height>` 是边界框的坐标，而 `<px1> <py1> <px2> <py2> ... <pxn> <pyn>` 是关键点的像素坐标。这些坐标之间用空格分隔。

    关于`<visibility>`选项，参考[官方给的回答](https://github.com/ultralytics/ultralytics/issues/6945)，共有三种值可选择：

    - `0`：表示关键点不可见。
    - `1`：表示关键点可见，但可能被遮挡。
    - `2`：表示关键点可见，且没有被遮挡。

可见如果需要训练`yolo-pose`模型，需要标注物体的矩形边界框和关键点，还要知道边界框对应的是哪些关键点，以及关键点的visibility。因此在介绍如何使用本转换工具前，需要约定好`labelme`的标注方式，确保形成的标注文件包含这些信息。

我的标注策略是使用矩形标注物体边界，用点标注关键点。额外给关键点添加`labelflag`属性，当关键点被遮挡时，激活该属性，那么`<visibility>`置1；当关键点不被遮挡时，不激活该属性，那么`<visibility>`置2；如果关键点被遮挡，那么就会生成空的标注信息，`<visibility>`置0。比如说我要标注下面这个柚子：

![recording](./assets/recording.gif)

对于中间的柚子，需要标注它的头和尾，它的尾部没有被遮挡，选择`tail`标签即可。而它的头部被叶子挡住了，因此标注的同时还要额外把下面的`occlude`标签勾选上；此外，还需要给它们分配同一个`group ID`（这个例子中为6，如果只有一个物体的话不用`group ID`也可以），告诉计算机这个柚子的头和尾对应的是哪个点，避免和其他柚子的混淆。

那么要如何设置`labelflag`呢？在启动`labelme`时添加`--labelflags`参数即可：

```bash
labelme --labelflags <path-to-labelflags-file> --labels <path-to-label-file>
```

其中的`<path-to-labelflags-file>`是`labelflags`的`yaml`配置文件的路径，`<path-to-label-file>`是标签的`txt`配置文件路径。

上述例子中，我用的标签有：`pomelo`,  `head`, `tail`，其中`head`和`tail`有`occlude`属性，配置文件可以写成：

```yaml
# labelflags.yaml
head:
- occlude
tail:
- occlude
```

```txt
# labels.txt
pomelo
head
tail
```

然后可以运行上面的命令进行标注了。

## 格式转换

### 安装依赖

```bash
pip install numpy
pip install pandas
pip install tqdm
```

## 运行

首先切换到项目目录下，然后运行下面的命令：

```bash
python convert.py --input <path-to-json-directory> --output <path-to-output-directory> --cfg_label <path-to-labels.txt>
```

这样应该就没问题了。

在`/experiment`文件夹中有一些样品可供测试。
