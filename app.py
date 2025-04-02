import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# 配置
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 限制上传大小为2MB

# 类别名称映射 (GTSRB的43个类别)
CLASS_NAMES = {
    0: "限速20公里/小时",
    1: "限速30公里/小时",
    2: "限速50公里/小时",
    3: "限速60公里/小时",
    4: "限速70公里/小时",
    5: "限速80公里/小时",
    6: "解除限速80公里/小时",
    7: "限速100公里/小时",
    8: "限速120公里/小时",
    9: "禁止超车",
    10: "禁止货车超车",
    11: "优先道路",
    12: "让行",
    13: "停车",
    14: "禁止车辆通行",
    15: "禁止货车通行",
    16: "禁止通行",
    17: "注意危险",
    18: "左急转弯",
    19: "右急转弯",
    20: "连续弯路",
    21: "不平路面",
    22: "湿滑路面",
    23: "路面变窄",
    24: "施工",
    25: "交通信号灯",
    26: "注意行人",
    27: "注意儿童",
    28: "注意自行车",
    29: "注意雪/冰",
    30: "注意野生动物",
    31: "解除所有限速和禁止",
    32: "右转",
    33: "左转",
    34: "直行",
    35: "直行或右转",
    36: "直行或左转",
    37: "靠右行驶",
    38: "靠左行驶",
    39: "环岛",
    40: "解除禁止超车",
    41: "解除禁止货车超车",
    42: "其他危险"
}

# 加载模型
model = load_model('traffic_sign_model.h5')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path):
    """
    预处理上传的图像，使其符合模型输入要求
    """
    img = Image.open(image_path)
    img = img.resize((32, 32))
    img_array = np.array(img) / 255.0

    # 处理不同通道数的图像
    if len(img_array.shape) == 2:  # 灰度图转RGB
        img_array = np.stack((img_array,) * 3, axis=-1)
    elif img_array.shape[2] == 4:  # RGBA转RGB
        img_array = img_array[:, :, :3]

    return np.expand_dims(img_array, axis=0)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # 检查是否有文件上传
        if 'file' not in request.files:
            return render_template('upload.html', error="请选择文件")

        file = request.files['file']

        # 检查文件名是否为空
        if file.filename == '':
            return render_template('upload.html', error="请选择文件")

        # 检查文件类型
        if not allowed_file(file.filename):
            return render_template('upload.html', error="仅支持PNG、JPG、JPEG格式")

        if file and allowed_file(file.filename):
            # 安全保存文件
            filename = secure_filename(file.filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                # 预处理图像并进行预测
                processed_img = preprocess_image(filepath)
                predictions = model.predict(processed_img)
                predicted_class = np.argmax(predictions[0])
                confidence = np.max(predictions[0])

                # 获取类别名称
                class_name = CLASS_NAMES.get(predicted_class, "未知交通标志")

                return render_template('result.html',
                                       image_path=os.path.join('uploads', filename),
                                       class_name=class_name,
                                       confidence=f"{confidence * 100:.2f}%",
                                       class_id=predicted_class)
            except Exception as e:
                return render_template('upload.html', error=f"处理图像时出错: {str(e)}")

    return render_template('upload.html')


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)