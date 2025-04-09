import tensorflow as tf

# Load lại model đã lưu
model = tf.keras.models.load_model("bird_classifier_mobilenetv2_final.keras")

# Tạo converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# (Tùy chọn) Bật tối ưu hóa để giảm kích thước
# converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Chuyển sang định dạng TFLite
tflite_model = converter.convert()

# Ghi ra file .tflite
with open("bird_classifier_mobilenetv2_final.tflite", "wb") as f:
    f.write(tflite_model)
