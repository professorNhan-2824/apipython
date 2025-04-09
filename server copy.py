from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# Tải mô hình Keras từ file .keras
model = load_model(r'F:\Python\bird_classifier_mobilenetv2_final.keras')  # Thay bằng đường dẫn file .keras của bạn

# Danh sách các loài chim (phải khớp với thứ tự lớp khi bạn train mô hình)
bird_classes = [
    'Đớp ruồi bụng vàng', 'Chim ruồi họng đỏ', 'Chim ruồi hung', 'Giẻ cùi lam',
    'Chích tối mắt', 'Bói cá mào', 'Bói cá bụng trắng', 'Vịt cổ xanh',
    'Vịt Merganser ngực đỏ', 'Quạ đen thường', 'Chim sẻ nhà', 'Chích vàng',
    'Chim bách thanh tuyết tùng', 'Gõ kiến đầu đỏ'
]

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    # Đọc ảnh từ request
    image_file = request.files['image']
    image = Image.open(image_file).resize((224, 224)).convert('RGB')  # Giả sử input là 224x224
    image_array = np.array(image, dtype=np.float32) / 255.0  # Chuẩn hóa về [0, 1]
    input_data = np.expand_dims(image_array, axis=0)  # Thêm batch dimension

    # Dự đoán bằng mô hình Keras
    
    predictions = model.predict(input_data)[0]
    max_index = np.argmax(predictions)
    confidence = float(predictions[max_index]) * 100

    # Trả về kết quả
    result = {
        'bird': bird_classes[max_index],
        'confidence': f'{confidence:.2f}%'
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)