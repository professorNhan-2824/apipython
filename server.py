from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import os

app = Flask(__name__)
CORS(app)


# Sử dụng đường dẫn tuyệt đối cho model
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'bird_classifier_mobilenetv2_final.tflite')

try:
    # Kiểm tra xem file có tồn tại không
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}")
        interpreter = None
    else:
        # Sử dụng TensorFlow Lite interpreter
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # Lấy thông tin về input và output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    interpreter = None

# Phần còn lại của code giữ nguyên

# Danh sách lớp chim
bird_classes = [
    'Đớp ruồi bụng vàng', 'Chim ruồi họng đỏ', 'Chim ruồi hung', 'Giẻ cùi lam',
    'Chích tối mắt', 'Bói cá mào', 'Bói cá bụng trắng', 'Vịt cổ xanh',
    'Vịt Merganser ngực đỏ', 'Quạ đen thường', 'Chim sẻ nhà', 'Chích vàng',
    'Chim bách thanh tuyết tùng', 'Gõ kiến đầu đỏ'
]

@app.route('/predict', methods=['POST'])
def predict():
    # Kiểm tra xem có file ảnh trong request không
    if 'image' not in request.files:
        return jsonify({'error': 'Không có ảnh được gửi lên'}), 400

    try:
        # Đọc ảnh từ request
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'File ảnh không hợp lệ'}), 400

        # Mở và xử lý ảnh
        image = Image.open(image_file).convert('RGB')
        image = image.resize((224, 224))  # Resize đúng input của model

        # Tiền xử lý ảnh
        image_array = np.array(image, dtype=np.float32) / 255.0  # Normalize
        input_tensor = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Kiểm tra interpreter
        if interpreter is None:
            return jsonify({'error': 'Mô hình không được tải'}), 500

        # Dự đoán sử dụng TFLite interpreter
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        
        max_index = int(np.argmax(predictions))
        confidence = float(predictions[max_index]) * 100

        result = {
            'bird': bird_classes[max_index],
            'confidence': f'{confidence:.2f}%'
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'Lỗi xử lý ảnh: {str(e)}'}), 500

if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5000, debug=True)

