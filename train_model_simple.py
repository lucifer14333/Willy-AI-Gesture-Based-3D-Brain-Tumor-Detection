"""
SIMPLIFIED BRAIN TUMOR MODEL TRAINING
Handles dependency issues gracefully
"""

import os
import sys

print("\n" + "=" * 70)
print("  BRAIN TUMOR AI MODEL TRAINING (SIMPLIFIED)")
print("=" * 70)

# Check dependencies
print("\n🔍 Checking dependencies...")

try:
    import numpy as np
    print(f"✅ NumPy {np.__version__}")
except ImportError:
    print("❌ NumPy not found. Install: pip install numpy<2.0")
    sys.exit(1)

try:
    import tensorflow as tf
    print(f"✅ TensorFlow {tf.__version__}")
except ImportError:
    print("❌ TensorFlow not found. Install: pip install tensorflow==2.15.0")
    sys.exit(1)

try:
    from tensorflow import keras
    print(f"✅ Keras ready")
except ImportError:
    print("❌ Keras not found")
    sys.exit(1)

try:
    import PIL
    print(f"✅ Pillow ready")
except ImportError:
    print("❌ Pillow not found. Install: pip install Pillow")
    sys.exit(1)

print("\n✅ All dependencies OK!")

# Configuration
DATASET_PATH = "./brain_mri_dataset"
IMG_SIZE = (128, 128)
BATCH_SIZE = 16  # Reduced for stability
EPOCHS = 15
MODEL_SAVE_PATH = "brain_tumor_model_lite.h5"

print(f"\n📁 Dataset path: {DATASET_PATH}")
print(f"   Image size: {IMG_SIZE}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Epochs: {EPOCHS}")

# Check if dataset exists
if not os.path.exists(DATASET_PATH):
    print("\n" + "=" * 70)
    print("❌ ERROR: Dataset not found!")
    print("=" * 70)
    print(f"\nExpected location: {os.path.abspath(DATASET_PATH)}")
    print("\n📥 Download the dataset from Kaggle:")
    print("   https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection")
    print("\n📦 Extract structure should be:")
    print("   brain_mri_dataset/")
    print("       yes/  (tumor images)")
    print("       no/   (no tumor images)")
    print("\n" + "=" * 70)
    sys.exit(1)

# Check folder structure
yes_path = os.path.join(DATASET_PATH, "yes")
no_path = os.path.join(DATASET_PATH, "no")

if not os.path.exists(yes_path) or not os.path.exists(no_path):
    print("\n❌ ERROR: Incorrect dataset structure!")
    print(f"\n   Missing folders:")
    if not os.path.exists(yes_path):
        print(f"   ❌ {yes_path}")
    if not os.path.exists(no_path):
        print(f"   ❌ {no_path}")
    print("\n   Required structure:")
    print("   brain_mri_dataset/")
    print("       yes/  (tumor images)")
    print("       no/   (no tumor images)")
    sys.exit(1)

# Count images
yes_count = len([f for f in os.listdir(yes_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
no_count = len([f for f in os.listdir(no_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

print(f"\n✅ Dataset found!")
print(f"   Tumor images (yes): {yes_count}")
print(f"   Normal images (no): {no_count}")
print(f"   Total images: {yes_count + no_count}")

if yes_count == 0 or no_count == 0:
    print("\n❌ ERROR: No images found in folders!")
    sys.exit(1)

# Data preparation
print("\n📊 Preparing data generators...")

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

print("   Creating training generator...")
train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

print("   Creating validation generator...")
val_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

print(f"✅ Data ready!")
print(f"   Training samples: {train_generator.samples}")
print(f"   Validation samples: {val_generator.samples}")

# Build model
print("\n🏗️  Building lightweight model...")

base_model = keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights='imagenet',
    alpha=0.35
)

base_model.trainable = False

model = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\n📐 Model architecture:")
model.summary()

# Training
print(f"\n🎓 Training for {EPOCHS} epochs...")
print("   This will take 10-15 minutes...")
print("=" * 70)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

try:
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    
    # Evaluation
    print("\n🔍 Evaluating model...")
    results = model.evaluate(val_generator, verbose=0)
    
    print(f"\n📊 Final Results:")
    print(f"   Loss: {results[0]:.4f}")
    print(f"   Accuracy: {results[1]:.4f} ({results[1]*100:.1f}%)")
    
    # Save model
    print(f"\n💾 Saving model...")
    model.save(MODEL_SAVE_PATH)
    
    file_size = os.path.getsize(MODEL_SAVE_PATH) / (1024 * 1024)
    print(f"✅ Model saved: {MODEL_SAVE_PATH} ({file_size:.2f} MB)")
    
    # Convert to TFLite
    print("\n🔄 Converting to TensorFlow Lite (for IoT)...")
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        tflite_path = MODEL_SAVE_PATH.replace('.h5', '.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        tflite_size = os.path.getsize(tflite_path) / (1024 * 1024)
        print(f"✅ TFLite model saved: {tflite_path} ({tflite_size:.2f} MB)")
        print(f"   Perfect for Jetson Nano!")
    except Exception as e:
        print(f"⚠️  TFLite conversion failed: {e}")
        print("   (H5 model is still available)")
    
    # Plot training history (optional)
    try:
        import matplotlib.pyplot as plt
        
        print("\n📊 Generating training plots...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy
        ax1.plot(history.history['accuracy'], label='Training')
        ax1.plot(history.history['val_accuracy'], label='Validation')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss
        ax2.plot(history.history['loss'], label='Training')
        ax2.plot(history.history['val_loss'], label='Validation')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150)
        print("✅ Plot saved: training_history.png")
    except ImportError:
        print("⚠️  Matplotlib not available - skipping plots")
    except Exception as e:
        print(f"⚠️  Plot generation failed: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("🎉 SUCCESS! Training completed successfully!")
    print("=" * 70)
    print("\n📦 Generated files:")
    print(f"   • {MODEL_SAVE_PATH} - Main model ({file_size:.1f} MB)")
    if os.path.exists(MODEL_SAVE_PATH.replace('.h5', '.tflite')):
        print(f"   • {MODEL_SAVE_PATH.replace('.h5', '.tflite')} - IoT version")
    print("   • best_model.h5 - Best checkpoint")
    if os.path.exists('training_history.png'):
        print("   • training_history.png - Training graphs")
    
    print("\n🚀 Next steps:")
    print("   1. Check training_history.png for performance")
    print(f"   2. Use {MODEL_SAVE_PATH} in brain_tumor_3d_simple.py")
    print("   3. Deploy to Jetson Nano if needed")
    print("\n" + "=" * 70)

except KeyboardInterrupt:
    print("\n\n⚠️  Training interrupted by user!")
    print("   Partial model may be saved as best_model.h5")
    sys.exit(1)
    
except Exception as e:
    print(f"\n\n❌ ERROR during training: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)