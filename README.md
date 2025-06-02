# Emotion Recognition Program / 情绪识别程序

This is a real-time emotion recognition Python program based on a webcam. It utilizes `opencv-python` for video capture and display, and the `deepface` library (built on TensorFlow/Keras) for face detection and emotion recognition.

这是一个基于摄像头进行实时情绪识别的 Python 程序。它利用 `opencv-python` 进行视频捕获和显示，以及 `deepface` 库（基于 TensorFlow/Keras）进行人脸检测和情绪识别。

## Features / 特性

*   **Real-time Emotion Recognition**: Captures video stream from the webcam in real-time and performs emotion recognition on detected faces.
    **实时情绪识别**: 从摄像头实时捕获视频流，并对检测到的人脸进行情绪识别。
*   **Multi-emotion Support**: Can identify multiple emotions (e.g., angry, disgust, fear, happy, sad, surprise, neutral).
    **多情绪支持**: 可以识别多种情绪（例如：生气、厌恶、恐惧、开心、悲伤、惊讶、中性）。
*   **Intuitive Display**: Draws bounding boxes around detected faces on the video frame and displays the identified emotion and its confidence score above the box.
    **直观显示**: 在视频帧上绘制人脸矩形框，并在框上方显示识别出的情绪及其置信度。
*   **Bilingual Emotion Display**: Supports displaying emotion names in both English and Chinese, configurable in `emotion_detector.py`.
    **中英双语情绪显示**: 支持以英文和中文显示情绪名称，可在 `emotion_detector.py` 中配置。

## Accuracy Notes / 准确率说明

The accuracy of emotion recognition in this program primarily relies on the pre-trained models used within the `deepface` library. `deepface` typically employs state-of-the-art deep learning models for emotion recognition and generally performs well on most standard datasets. `deepface` integrates various face detection and emotion recognition models internally, and its `DeepFace.analyze` function can select different models as needed. By default, it attempts to use well-performing models.

本程序的情绪识别准确率主要依赖于 `deepface` 库中使用的预训练模型。`deepface` 库通常采用最新的深度学习模型进行情绪识别，在大多数标准数据集上表现良好。`deepface` 内部集成了多种人脸检测模型和情绪识别模型，其 `DeepFace.analyze` 函数可以根据需要选择不同的模型。默认情况下，它会尝试使用表现较好的模型。

Emotion recognition accuracy can be affected by various factors, including:
情绪识别的准确性可能受多种因素影响，包括：

*   **Lighting Conditions**: Poor or complex lighting can affect the accuracy of face detection and emotion recognition.
    **光照条件**: 弱光或复杂光照条件可能影响人脸检测和情绪识别的准确性。
*   **Facial Occlusion**: Obstructions such as glasses, masks, hands, or other objects on the face can reduce accuracy.
    **面部遮挡**: 眼镜、口罩、手或其他物体对面部的遮挡会降低准确率。
*   **Head Pose and Exaggeration of Expressions**: Extreme head poses or atypical facial expressions can lead to recognition errors.
    **头部姿态和表情夸张度**: 极端头部姿态或非典型的面部表情可能导致识别错误。
*   **Model Generalization Capability**: Although pre-trained models are trained on large datasets, their performance may decrease when encountering unfamiliar faces or expressions in real-world scenarios.
    **模型泛化能力**: 尽管预训练模型在大量数据上进行了训练，但在面对未见过的人脸或表情时，其性能可能会有所下降。

To improve face detection accuracy, this program uses the `retinaface` backend for `DeepFace.analyze`. We set `enforce_detection=False` in the `DeepFace.analyze` function, which means the program will not immediately throw an error if no face is detected, but will continue to try processing the next frame, making it more user-friendly in real-time applications.

为了提高人脸检测的准确性，本程序在 `DeepFace.analyze` 函数中使用了 `retinaface` 后端。我们设置了 `enforce_detection=False`，这意味着即使未检测到人脸，程序也不会立即抛出错误，而是继续尝试处理下一帧，这在实时应用中更友好。

## Installation and Running / 安装与运行

### 1. Clone the Project / 克隆项目

If you haven't cloned this project yet, please do so:
如果您尚未克隆此项目，请先进行克隆：

```bash
# git clone [Project URL]
# cd [Project Name]
```

### 2. Create and Activate Conda Virtual Environment (Recommended) / 创建并激活 Conda 虚拟环境 (推荐)

To avoid dependency conflicts, it is highly recommended to use a Conda virtual environment. This program requires Python 3.12.
为了避免依赖冲突，强烈建议使用 Conda 虚拟环境。本程序需要 Python 3.12 版本。

```bash
conda create --name emotion_env python=3.12 -y
conda activate emotion_env
```

### 3. Install Dependencies / 安装依赖

After activating the Conda environment, install all dependencies listed in `requirements.txt`:
在激活 Conda 环境后，安装 `requirements.txt` 中列出的所有依赖：

```bash
pip install -r requirements.txt
```

**Note**: Installing `tensorflow` and `deepface` may take some time, and `deepface` will automatically download its required model files (usually stored in the `.deepface` folder under your user's home directory) the first time it runs. Please ensure a stable network connection. If your system has an NVIDIA GPU, you can install `tensorflow-gpu` for better performance.

**注意**: 安装 `tensorflow` 和 `deepface` 可能需要较长时间，并且 `deepface` 在第一次运行时会自动下载其所需的模型文件（通常存储在用户主目录下的 `.deepface` 文件夹中），请确保网络连接畅通。如果您的系统有 NVIDIA GPU，可以安装 `tensorflow-gpu` 以获得更好的性能。

### 4. Run the Program / 运行程序

After installing all dependencies, you can run the emotion recognition program:
安装所有依赖后，您可以运行情绪识别程序：

```bash
python emotion_detector.py
```

Upon running, you should see a pop-up window displaying a real-time video stream from your default webcam, with identified emotions displayed above detected faces.
运行后，您应该会看到一个弹出窗口，显示来自您默认摄像头的实时视频流，并在检测到的人脸上方显示识别出的情绪。

### 5. Exit the Program / 退出程序

In the video window, press the `q` key on your keyboard to exit the program.
在视频窗口中，按键盘上的 `q` 键即可退出程序。

## Code Specification / 代码规范

This program adheres to the following basic code specifications:
本程序遵循以下基本代码规范：

*   **PEP 8**: Follows Python's official PEP 8 style guide, including naming conventions, indentation, blank lines, etc.
    **PEP 8**: 遵循 Python 官方的 PEP 8 风格指南，包括命名约定、缩进、空行等。
*   **Comments**: Key code logic and functions are accompanied by clear comments for easy understanding.
    **注释**: 关键代码逻辑和函数都配有清晰的注释，方便理解。
*   **Function and Variable Naming**: Variable and function names aim to express their purpose, improving code readability.
    **函数和变量命名**: 变量和函数名力求表达其用途，提高代码可读性。
*   **Error Handling**: Includes basic camera opening error handling and `deepface` analysis error handling.
    **错误处理**: 包含基本的摄像头打开错误处理和 `deepface` 分析错误处理。

## License / 许可证

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

本项目采用 MIT 许可证 - 有关详细信息，请参阅 [LICENSE.md](LICENSE.md) 文件。 