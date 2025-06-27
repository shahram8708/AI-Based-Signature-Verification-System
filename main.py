!pip install gradio tensorflow opencv-python-headless pillow numpy matplotlib scikit-learn

import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import gradio as gr
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings('ignore')

print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

def create_synthetic_signature_data(num_people=50, signatures_per_person=10):
    print("Creating synthetic signature dataset...")
    
    os.makedirs('signature_data/genuine', exist_ok=True)
    os.makedirs('signature_data/forged', exist_ok=True)
    
    genuine_pairs = []
    forged_pairs = []
    
    for person_id in range(num_people):
        person_signatures = []
        
        for sig_id in range(signatures_per_person):
            img = np.ones((150, 150), dtype=np.uint8) * 255  
            
            num_strokes = random.randint(3, 8)
            for _ in range(num_strokes):
                start_x = random.randint(10, 140)
                start_y = random.randint(30, 120)
                end_x = start_x + random.randint(-30, 30)
                end_y = start_y + random.randint(-20, 20)
                
                cv2.line(img, (start_x, start_y), (end_x, end_y), 0, 
                        thickness=random.randint(1, 3))
                
                if random.random() > 0.5:
                    center_x = (start_x + end_x) // 2
                    center_y = (start_y + end_y) // 2
                    radius = random.randint(5, 15)
                    cv2.circle(img, (center_x, center_y), radius, 0, 
                             thickness=random.randint(1, 2))
            
            person_signatures.append(img)
        
        for i in range(len(person_signatures)):
            for j in range(i+1, len(person_signatures)):
                genuine_pairs.append((person_signatures[i], person_signatures[j], 1))
        
        if person_id == 0:
            all_signatures = person_signatures.copy()
            person_labels = [person_id] * len(person_signatures)
        else:
            all_signatures.extend(person_signatures)
            person_labels.extend([person_id] * len(person_signatures))
    
    print(f"Generated {len(genuine_pairs)} genuine pairs")
    
    for _ in range(len(genuine_pairs)):
        idx1, idx2 = random.sample(range(len(all_signatures)), 2)
        while person_labels[idx1] == person_labels[idx2]:
            idx1, idx2 = random.sample(range(len(all_signatures)), 2)
        forged_pairs.append((all_signatures[idx1], all_signatures[idx2], 0))
    
    print(f"Generated {len(forged_pairs)} forged pairs")
    
    return genuine_pairs + forged_pairs

def preprocess_signature(image):
    if isinstance(image, str):
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    elif isinstance(image, np.ndarray):
        if len(image.shape) == 3:
            img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            img = image.copy()
    else:
        img = np.array(image)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    if img is None:
        raise ValueError("Could not load image")
    
    img = cv2.resize(img, (150, 150))
    
    img = cv2.GaussianBlur(img, (3, 3), 0)
    
    img = img.astype(np.float32) / 255.0
    
    img = np.expand_dims(img, axis=-1)
    
    return img

def augment_signature(image):
    if len(image.shape) == 3 and image.shape[-1] == 1:
        img = image.squeeze()  
        was_3d = True
    else:
        img = image.copy()
        was_3d = False
    
    if img.dtype == np.float32:
        img_uint8 = (img * 255).astype(np.uint8)
    else:
        img_uint8 = img.astype(np.uint8)
    
    if random.random() > 0.5:
        angle = random.uniform(-10, 10)
        center = (75, 75)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        img_uint8 = cv2.warpAffine(img_uint8, rotation_matrix, (150, 150))
    
    if random.random() > 0.5:
        scale_factor = random.uniform(0.9, 1.1)
        new_size = int(150 * scale_factor)
        img_temp = cv2.resize(img_uint8, (new_size, new_size))
        img_uint8 = cv2.resize(img_temp, (150, 150))
    
    img_float = img_uint8.astype(np.float32) / 255.0
    
    if len(img_float.shape) == 2:
        img_float = np.expand_dims(img_float, axis=-1)
    
    assert img_float.shape == (150, 150, 1), f"Augmentation output shape error: {img_float.shape}"
    
    return img_float

def create_base_network(input_shape):
    input_layer = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    output = layers.Dense(128, activation='relu')(x)
    
    return Model(input_layer, output)

def create_siamese_network(input_shape):
    base_network = create_base_network(input_shape)
    
    input_a = layers.Input(shape=input_shape)
    input_b = layers.Input(shape=input_shape)
    
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    
    difference = layers.Lambda(lambda x: tf.abs(x[0] - x[1]))([processed_a, processed_b])
    
    similarity = layers.Dense(64, activation='relu')(difference)
    similarity = layers.Dropout(0.5)(similarity)
    similarity = layers.Dense(32, activation='relu')(similarity)
    similarity = layers.Dropout(0.5)(similarity)
    output = layers.Dense(1, activation='sigmoid')(similarity)
    
    model = Model(inputs=[input_a, input_b], outputs=output)
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def prepare_training_data(signature_pairs):
    X1, X2, y = [], [], []
    
    print(f"Processing {len(signature_pairs)} signature pairs...")
    successful_pairs = 0
    
    for i, pair in enumerate(signature_pairs):
        try:
            img1, img2, label = pair
            
            img1_processed = preprocess_signature(img1)
            img2_processed = preprocess_signature(img2)
            
            assert img1_processed.shape == (150, 150, 1), f"Image 1 preprocessing failed: {img1_processed.shape}"
            assert img2_processed.shape == (150, 150, 1), f"Image 2 preprocessing failed: {img2_processed.shape}"
            
            if random.random() > 0.5:
                img1_processed = augment_signature(img1_processed)
            if random.random() > 0.5:
                img2_processed = augment_signature(img2_processed)
            
            assert img1_processed.shape == (150, 150, 1), f"Image 1 final shape error: {img1_processed.shape}"
            assert img2_processed.shape == (150, 150, 1), f"Image 2 final shape error: {img2_processed.shape}"
            
            X1.append(img1_processed)
            X2.append(img2_processed)
            y.append(label)
            successful_pairs += 1
            
            if (i + 1) % 200 == 0:
                print(f"Successfully processed {successful_pairs}/{i + 1} pairs")
                
        except Exception as e:
            print(f"Error processing pair {i}: {e}")
            continue
    
    print(f"Successfully processed {successful_pairs} out of {len(signature_pairs)} pairs")
    
    try:
        X1_array = np.array(X1, dtype=np.float32)
        X2_array = np.array(X2, dtype=np.float32)
        y_array = np.array(y, dtype=np.float32)
        
        print(f"✅ Arrays created successfully!")
        print(f"Final shapes: X1={X1_array.shape}, X2={X2_array.shape}, y={y_array.shape}")
        
        return X1_array, X2_array, y_array
        
    except Exception as e:
        print(f"❌ Error creating arrays: {e}")
        print("Debugging information:")
        if X1:
            print(f"Sample X1 shapes: {[x.shape for x in X1[:5]]}")
        if X2:
            print(f"Sample X2 shapes: {[x.shape for x in X2[:5]]}")
        raise

def train_model():
    print("Starting model training...")
    
    try:
        signature_pairs = create_synthetic_signature_data(num_people=25, signatures_per_person=6)
        
        X1, X2, y = prepare_training_data(signature_pairs)
        
        print(f"Training data prepared successfully!")
        print(f"Data shapes: X1={X1.shape}, X2={X2.shape}, y={y.shape}")
        
        X1_train, X1_val, X2_train, X2_val, y_train, y_val = train_test_split(
            X1, X2, y, test_size=0.2, random_state=42, stratify=y
        )
        
        input_shape = (150, 150, 1)
        model = create_siamese_network(input_shape)
        
        print("Model architecture:")
        model.summary()
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7),
            tf.keras.callbacks.ModelCheckpoint('best_signature_model.h5', save_best_only=True)
        ]
        
        history = model.fit(
            [X1_train, X2_train], y_train,
            validation_data=([X1_val, X2_val], y_val),
            epochs=30,
            batch_size=16,
            callbacks=callbacks,
            verbose=1
        )
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        return model
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise

def compare_signatures(model, image1, image2, threshold=0.5):
    try:
        img1_processed = preprocess_signature(image1)
        img2_processed = preprocess_signature(image2)
        
        img1_batch = np.expand_dims(img1_processed, axis=0)
        img2_batch = np.expand_dims(img2_processed, axis=0)
        
        similarity_score = model.predict([img1_batch, img2_batch], verbose=0)[0][0]
        
        is_match = similarity_score > threshold
        confidence = float(similarity_score)
        
        result = {
            'match': is_match,
            'confidence': confidence,
            'similarity_score': confidence,
            'result_text': "Match" if is_match else "No Match",
            'confidence_text': f"Confidence: {confidence:.2%}"
        }
        
        return result
        
    except Exception as e:
        print(f"Error in signature comparison: {str(e)}")
        return {
            'match': False,
            'confidence': 0.0,
            'similarity_score': 0.0,
            'result_text': "Error in processing",
            'confidence_text': "Error occurred"
        }

def signature_verification_interface(signature1, signature2):
    if signature1 is None or signature2 is None:
        return "Please upload both signature images", "No confidence score available"
    
    try:
        global trained_model
        
        result = compare_signatures(trained_model, signature1, signature2)
        
        result_text = f"Result: {result['result_text']}"
        confidence_text = f"Confidence: {result['confidence']:.2%}"
        
        if result['confidence'] > 0.8:
            interpretation = "High confidence"
        elif result['confidence'] > 0.6:
            interpretation = "Medium confidence"
        elif result['confidence'] > 0.4:
            interpretation = "Low confidence"
        else:
            interpretation = "Very low confidence"
        
        full_result = f"{result_text}\n{confidence_text}\n{interpretation}"
        
        return full_result, f"Similarity Score: {result['similarity_score']:.4f}"
        
    except Exception as e:
        return f"Error: {str(e)}", "Error in processing"

def create_gradio_interface():
    with gr.Blocks(title="Signature Verification System") as demo:
        gr.Markdown("# 🖋️ AI-Powered Signature Verification System")
        gr.Markdown("Upload two signature images to verify if they belong to the same person")
        
        with gr.Row():
            with gr.Column():
                signature1_input = gr.Image(
                    label="Signature 1",
                    type="pil",
                    height=300
                )
            with gr.Column():
                signature2_input = gr.Image(
                    label="Signature 2", 
                    type="pil",
                    height=300
                )
        
        with gr.Row():
            verify_button = gr.Button("Verify Signatures", variant="primary", size="lg")
        
        with gr.Row():
            with gr.Column():
                result_output = gr.Textbox(
                    label="Verification Result",
                    lines=3,
                    interactive=False
                )
            with gr.Column():
                confidence_output = gr.Textbox(
                    label="Technical Details",
                    lines=3,
                    interactive=False
                )
        
        verify_button.click(
            fn=signature_verification_interface,
            inputs=[signature1_input, signature2_input],
            outputs=[result_output, confidence_output]
        )
        
        gr.Markdown("## 📝 Instructions")
        gr.Markdown("""
        1. Upload the first signature image using the 'Signature 1' upload area
        2. Upload the second signature image using the 'Signature 2' upload area
        3. Click 'Verify Signatures' to analyze the images
        4. The system will output whether the signatures match and confidence level
        
        **Tips for best results:**
        - Use clear, high-resolution images
        - Ensure signatures are clearly visible
        - Avoid blurry or low-quality images
        - White background works best
        """)
    
    return demo

def main():
    print("🖋️ Signature Verification System Starting...")
    print("=" * 50)
    
    try:
        if not os.path.exists('best_signature_model.h5'):
            print("Training new model...")
            global trained_model
            trained_model = train_model()
            print("✅ Model training completed!")
        else:
            print("Loading existing model...")
            input_shape = (150, 150, 1)
            trained_model = create_siamese_network(input_shape)
            trained_model.load_weights('best_signature_model.h5')
            print("✅ Model loaded successfully!")
        
        globals()['trained_model'] = trained_model
        
        print("Creating Gradio interface...")
        demo = create_gradio_interface()
        
        print("Launching application...")
        print("=" * 50)
        
        demo.launch(
            share=True, 
            server_name="0.0.0.0",
            server_port=7860,
            show_api=False
        )
        
    except Exception as e:
        print(f"❌ Application failed to start: {e}")
        raise

if __name__ == "__main__":
    main()

print("🚀 Signature Verification System Ready!")
