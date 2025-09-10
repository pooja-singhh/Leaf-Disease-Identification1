# import os
# import random
# import numpy as np
# import tensorflow as tf

# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import DenseNet121
# from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# # =========================
# # 0) Reproducibility (optional)
# # =========================
# SEED = 42
# random.seed(SEED)
# np.random.seed(SEED)
# tf.random.set_seed(SEED)

# # =========================
# # 1) Configuration
# # =========================
# # Update these paths to YOUR split folders
# train_dir = "./Train"
# val_dir   = "./Valid"
# test_dir  = "./Test"

# img_size = 224          # DenseNet121 works well at 224x224
# batch_size = 32
# epochs_initial = 15     # Feature-extraction phase
# epochs_finetune = 15    # Fine-tuning phase
# lr_initial = 3e-4
# lr_finetune = 1e-5
# dropout_rate = 0.35

# # =========================
# # 2) Optional Mixed Precision for speed (GPU only)
# # =========================
# use_mixed_precision = False
# try:
#     from tensorflow.keras import mixed_precision
#     mixed_precision.set_global_policy("mixed_float16")
#     use_mixed_precision = True
#     print(">> Mixed precision enabled.")
# except Exception as e:
#     print(">> Mixed precision not enabled (CPU or unsupported GPU). Proceeding in float32.")

# # =========================
# # 3) Data Generators (with strong but safe augmentations)
# # =========================
# # NOTE: We use rescale=1/255. This is fine for DenseNet pretrained weights in practice.
# # If you prefer the exact DenseNet preprocess_input, you can swap rescale for:
# #   preprocessing_function=tf.keras.applications.densenet.preprocess_input
# # and remove rescale.

# train_datagen = ImageDataGenerator(
#     rescale=1.0/255,
#     rotation_range=35,
#     width_shift_range=0.10,
#     height_shift_range=0.10,
#     zoom_range=0.25,
#     shear_range=0.15,
#     horizontal_flip=True,
#     brightness_range=(0.8, 1.2),
#     fill_mode="nearest"
# )

# val_datagen = ImageDataGenerator(rescale=1.0/255)
# test_datagen = ImageDataGenerator(rescale=1.0/255)

# train_gen = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(img_size, img_size),
#     batch_size=batch_size,
#     class_mode="categorical",
#     shuffle=True,
#     seed=SEED
# )

# val_gen = val_datagen.flow_from_directory(
#     val_dir,
#     target_size=(img_size, img_size),
#     batch_size=batch_size,
#     class_mode="categorical",
#     shuffle=False
# )

# test_gen = test_datagen.flow_from_directory(
#     test_dir,
#     target_size=(img_size, img_size),
#     batch_size=batch_size,
#     class_mode="categorical",
#     shuffle=False
# )

# num_classes = len(train_gen.class_indices)
# print("\nClass indices:", train_gen.class_indices)

# # =========================
# # 4) Model: DenseNet121 backbone + custom head
# # =========================
# base_model = DenseNet121(
#     weights="imagenet",
#     include_top=False,
#     input_tensor=Input(shape=(img_size, img_size, 3))
# )
# base_model.trainable = False  # freeze for initial training

# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dropout(dropout_rate)(x)
# x = Dense(256, activation="relu")(x)
# x = Dropout(dropout_rate)(x)

# # If mixed precision is on, ensure the output is float32 for numerical stability
# output = Dense(num_classes, activation="softmax", dtype="float32" if use_mixed_precision else None)(x)

# model = Model(inputs=base_model.input, outputs=output)
# model.compile(optimizer=Adam(learning_rate=lr_initial),
#               loss="categorical_crossentropy",
#               metrics=["accuracy"])

# model.summary()

# # =========================
# # 5) Callbacks
# # =========================
# chk_path = "densenet121_crop_best.keras"
# callbacks_initial = [
#     ModelCheckpoint(chk_path, monitor="val_accuracy", save_best_only=True, verbose=1),
#     EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
#     ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)
# ]

# # =========================
# # 6) Initial Training (feature extraction)
# # =========================
# print("\n=== Initial Training: feature extraction (frozen backbone) ===")
# history_initial = model.fit(
#     train_gen,
#     validation_data=val_gen,
#     epochs=epochs_initial,
#     callbacks=callbacks_initial
# )

# print("\n=== Evaluating on TEST (after initial training) ===")
# test_loss, test_acc = model.evaluate(test_gen, verbose=1)
# print(f"Test Accuracy (Initial): {test_acc:.4f}  |  Test Loss: {test_loss:.4f}")

# # =========================
# # 7) Fine-tuning (unfreeze backbone with low LR)
# # =========================
# print("\n=== Fine-Tuning: unfreezing backbone ===")
# base_model.trainable = True

# # (Optional) Unfreeze last N blocks only for even more stability:
# # for layer in base_model.layers[:-100]:
# #     layer.trainable = False

# model.compile(optimizer=Adam(learning_rate=lr_finetune),
#               loss="categorical_crossentropy",
#               metrics=["accuracy"])

# callbacks_finetune = [
#     ModelCheckpoint("densenet121_crop_finetuned_best.keras", monitor="val_accuracy", save_best_only=True, verbose=1),
#     EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
#     ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1)
# ]

# history_ft = model.fit(
#     train_gen,
#     validation_data=val_gen,
#     epochs=epochs_finetune,
#     callbacks=callbacks_finetune
# )

# # Save final fine-tuned model
# model.save("densenet121_crop_finetuned_final.keras")
# print("\nSaved: densenet121_crop_finetuned_final.keras")

# # =========================
# # 8) Final Evaluation on TEST
# # =========================
# print("\n=== Final Evaluation on TEST (fine-tuned) ===")
# test_loss_ft, test_acc_ft = model.evaluate(test_gen, verbose=1)
# print(f"Test Accuracy (Fine-tuned): {test_acc_ft:.4f}  |  Test Loss: {test_loss_ft:.4f}")

# # =========================
# # 9) (Optional) Predict & show per-class accuracy
# # =========================
# # This block computes per-class accuracy without extra libs.
# from collections import defaultdict

# print("\n=== Per-class Accuracy (optional) ===")
# test_gen.reset()
# preds = model.predict(test_gen, verbose=1)
# y_pred = np.argmax(preds, axis=1)
# y_true = test_gen.classes
# class_indices = {v: k for k, v in test_gen.class_indices.items()}

# per_class_correct = defaultdict(int)
# per_class_total = defaultdict(int)
# for t, p in zip(y_true, y_pred):
#     per_class_total[t] += 1
#     if t == p:
#         per_class_correct[t] += 1

# for idx in sorted(per_class_total.keys()):
#     acc = per_class_correct[idx] / max(1, per_class_total[idx])
#     print(f"{class_indices[idx]}: {acc*100:.2f}%  ({per_class_correct[idx]}/{per_class_total[idx]})")

# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms, models
# from torch.utils.data import DataLoader
# import torch.nn.functional as F

# # ==============================
# # CONFIG
# # ==============================
# DATA_DIR = "./dataset"  # dataset folder (train/val/test subfolders)
# BATCH_SIZE = 32
# NUM_EPOCHS = 10
# LEARNING_RATE = 1e-4
# CKPT_PATH = "best_densenet121.pth"
# NUM_CLASSES = 15  # <-- change if needed

# # ==============================
# # DATA
# # ==============================
# train_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(10),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225]),
# ])

# test_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225]),
# ])

# train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
# val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=test_transform)
# test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=test_transform)

# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
# val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
# test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# # ==============================
# # MODEL
# # ==============================
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# model = models.densenet121(weights="IMAGENET1K_V1")
# num_ftrs = model.classifier.in_features
# model.classifier = nn.Linear(num_ftrs, NUM_CLASSES)
# model = model.to(device)

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# # ==============================
# # TRAIN FUNCTION
# # ==============================
# def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
#     best_acc = 0.0

#     for epoch in range(num_epochs):
#         # ---- Training ----
#         model.train()
#         running_loss, running_corrects = 0.0, 0

#         for inputs, labels in train_loader:
#             inputs, labels = inputs.to(device), labels.to(device)

#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             _, preds = torch.max(outputs, 1)

#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item() * inputs.size(0)
#             running_corrects += torch.sum(preds == labels.data)

#         epoch_loss = running_loss / len(train_loader.dataset)
#         epoch_acc = running_corrects.double() / len(train_loader.dataset)

#         # ---- Validation ----
#         model.eval()
#         val_corrects = 0

#         with torch.no_grad():
#             for inputs, labels in val_loader:
#                 inputs, labels = inputs.to(device), labels.to(device)
#                 outputs = model(inputs)
#                 _, preds = torch.max(outputs, 1)
#                 val_corrects += torch.sum(preds == labels.data)

#         val_acc = val_corrects.double() / len(val_loader.dataset)

#         print(f"Epoch {epoch+1}/{num_epochs} "
#               f"Train Loss: {epoch_loss:.4f} Train Acc: {epoch_acc:.2f} "
#               f"Val Acc: {val_acc:.2f}")

#         # ---- Save checkpoint ----
#         if val_acc > best_acc:
#             best_acc = val_acc
#             torch.save(model.state_dict(), CKPT_PATH)
#             print(f"✅ Saved new best model with val_acc: {val_acc:.2f}")

#     print("Training complete! Best Val Acc: {:.2f}".format(best_acc))


# # ==============================
# # TEST FUNCTION
# # ==============================
# def test_model(model, test_loader, criterion):
#     model.eval()
#     test_loss, corrects = 0.0, 0

#     with torch.no_grad():
#         for inputs, labels in test_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             _, preds = torch.max(outputs, 1)

#             test_loss += loss.item() * inputs.size(0)
#             corrects += torch.sum(preds == labels.data)

#     test_loss /= len(test_loader.dataset)
#     test_acc = corrects.double() / len(test_loader.dataset)

#     print(f"🔍 Test Loss: {test_loss:.4f} Test Acc: {test_acc:.2f}")


# # ==============================
# # MAIN
# # ==============================
# if __name__ == "__main__":
#     # ---- Train ----
#     train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS)

#     # ---- Load best checkpoint ----
#     if os.path.exists(CKPT_PATH):
#         model.load_state_dict(torch.load(CKPT_PATH, map_location=device, weights_only=True))
#         print("✅ Best checkpoint loaded for testing!")
#     else:
#         print(f"❌ No checkpoint found at {CKPT_PATH}, testing skipped.")
#         exit()

#     # ---- Test ----
#     test_model(model, test_loader, criterion)
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ===============================
# 1. Device Setup
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===============================
# 2. Data Preprocessing
# ===============================
data_dir = "./dataset"  # <-- keep subfolders: train, valid, test

# Image transforms
transform = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    "test": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

# Load datasets
image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), transform=transform[x])
    for x in ["train", "val", "test"]
}

# Create dataloaders
dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=32, shuffle=True)
    for x in ["train", "val", "test"]
}

# Number of classes
num_classes = len(image_datasets["train"].classes)
print(f"Detected {num_classes} classes.")

# ===============================
# 3. Model Setup (EfficientNet-B0)
# ===============================
model = models.efficientnet_b0(pretrained=True)

# Freeze feature extractor
for param in model.features.parameters():
    param.requires_grad = False

# Replace classifier head
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ===============================
# 4. Training Loop
# ===============================
epochs = 5  # increase if needed
best_acc = 0.0

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    print("-" * 20)

    for phase in ["train", "val"]:
        if phase == "train":
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == "train"):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == "train":
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(image_datasets[phase])
        epoch_acc = running_corrects.double() / len(image_datasets[phase])

        print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Save best model
        if phase == "val" and epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("✅ Model saved!")

print("\nTraining complete.")
print(f"Best validation accuracy: {best_acc:.4f}")

# ===============================
# 5. Testing
# ===============================
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

test_corrects = 0

with torch.no_grad():
    for inputs, labels in dataloaders["test"]:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        test_corrects += torch.sum(preds == labels.data)

test_acc = test_corrects.double() / len(image_datasets["test"])
print(f"\n🎯 Test Accuracy: {test_acc:.4f}")

# ===============================
# 6. Save Final Model
# ===============================
torch.save(model.state_dict(), "final_model.pth")
print("📁 Final model saved as final_model.pth")
