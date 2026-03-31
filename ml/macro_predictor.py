import cv2
import numpy as np
import onnxruntime as ort

from utils.resource_path import resource_path

class MacroExpressionPredictor:
    def __init__(self, model_path="models/macro_expression.onnx"):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        # 1. Load Model ONNX
        try:
            # Load model dengan provider yang sudah diset
            self.ort_session = ort.InferenceSession(resource_path(model_path), providers=providers)
            
            # --- DEBUGGING: Cek apakah GPU terbaca ---
            active_providers = self.ort_session.get_providers()
            print(f"Model loaded using: {active_providers[0]}") 
        except Exception as e:
            print(f"Error loading model: {e}")
            self.ort_session = None
            return

        # 2. Dapatkan info input/output layer
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name
        
        # 3. Definisikan Label (Sesuaikan urutan dengan training model Anda)
        # Urutan standar biasanya: [Neutral, Anger, Disgust, Fear, Happiness, Sadness, Surprise]
        self.labels = [
            "Negative", "Neutral", "Positive",
        ]
        # self.labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

    def preprocess(self, face_image):
        input_tensor = face_image.astype(np.float32)
        
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor

    def softmax(self, x):
        """Menghitung probabilitas dari logits"""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def predict(self, face_image):
        if self.ort_session is None or face_image is None:
            return "Error", 0.0

        # 1. Preprocess
        input_tensor = self.preprocess(face_image)

        # 2. Inference
        outputs = self.ort_session.run([self.output_name], {self.input_name: input_tensor})
        logits = outputs[0][0] # Ambil hasil batch pertama

        pred_idx = np.argmax(logits)
        confidence = logits[pred_idx]
        label = self.labels[pred_idx] if pred_idx < len(self.labels) else "Unknown"

        return label, float(confidence)