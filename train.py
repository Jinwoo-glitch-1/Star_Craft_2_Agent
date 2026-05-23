import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

# استدعاء الموديل من ملف lstm_model اللي عملناه في 4.1 و 4.2
from ai_models.lstm_model import ResourcePredictor, setup_training
# ==========================================
# 1. Dataset Class (تجهيز البيانات)
# ==========================================
class SC2Dataset(Dataset):
    def __init__(self, X_data, y_data):
        """
        بيستقبل مصفوفات الداتا ويجهزها لـ PyTorch
        X_data: (Samples, 60, 8)
        y_data: (Samples, 1)
        """
        self.X = torch.tensor(X_data, dtype=torch.float32)
        self.y = torch.tensor(y_data, dtype=torch.float32).view(-1, 1) # التأكد إن شكل الناتج (Batch, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==========================================
# 2. Training Loop (حلقة التدريب)
# ==========================================
def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    """
    TASK 4.3: Model Training (Offline Training)
    """
    print(f"--- Starting Training for {num_epochs} Epochs ---")
    
    # تحويل الموديل لوضع التدريب
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for batch_X, batch_y in dataloader:
            # 1. تصفير الـ Gradients (عشان القيم ماتتراكمش من الخطوة اللي قبلها)
            optimizer.zero_grad()
            
            # 2. التوقع (Forward Pass)
            predictions = model(batch_X)
            
            # 3. حساب نسبة الخطأ (Loss)
            loss = criterion(predictions, batch_y)
            
            # 4. حساب التعديلات المطلوبة في الأوزان (Backward Pass)
            loss.backward()
            
            # 5. تحديث الأوزان (Optimizer Step)
            optimizer.step()
            
            epoch_loss += loss.item()
            
        # طباعة متوسط الخطأ بعد كل Epoch
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss (MSE): {avg_loss:.4f}")
        
    print("--- Training Completed Successfully! ---")
    return model

# ==========================================
# 3. Save Model (حفظ الموديل)
# ==========================================
def save_model(model, filename="resource_predictor.pth"):
    os.makedirs("models", exist_ok=True)
    filepath = os.path.join("models", filename)
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath} ✅")

# ==========================================
# Entry Point (نقطة التشغيل)
# ==========================================
if __name__ == "__main__":
    # ---------------------------------------------------------
    # الجزء ده بيعمل داتا وهمية للـ Testing لحد ما التيم يخلص
    # ---------------------------------------------------------
    print("Generating Dummy Data for testing the pipeline...")
    num_samples = 1000  # 1000 موقف في اللعبة
    seq_length = 60     # دقيقة ورا بعض
    num_features = 8    # 8 فيتشرز (معادن، عمال، إلخ)
    
    # داتا وهمية للتدريب
    dummy_X = np.random.rand(num_samples, seq_length, num_features) 
    # أرقام وهمية لكمية الموارد اللي هتتسحب (مثلاً بين 0 و 1000 معدن)
    dummy_y = np.random.rand(num_samples) * 1000 
    
    # ---------------------------------------------------------
    # تشغيل الـ Pipeline بالكامل
    # ---------------------------------------------------------
    # 1. تجهيز الـ DataLoader (بيجيب الداتا حتة حتة، كل حتة 32 سطر)
    dataset = SC2Dataset(dummy_X, dummy_y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 2. استدعاء الموديل (تاسك 4.1)
    model = ResourcePredictor(input_size=num_features)
    
    # 3. استدعاء الـ Loss والـ Optimizer (تاسك 4.2)
    criterion, optimizer = setup_training(model, learning_rate=0.001)
    
    # 4. التدريب الفعلي (تاسك 4.3)
    trained_model = train_model(model, dataloader, criterion, optimizer, num_epochs=5)
    
    # 5. حفظ الموديل لاستخدامه جوه اللعبة
    save_model(trained_model)