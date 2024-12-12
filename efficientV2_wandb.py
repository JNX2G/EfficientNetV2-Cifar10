import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from time import time
from torch.cuda.amp import GradScaler, autocast
import wandb

#%% path and parameters
base_path = 'C:/workspace/SKKU_DS/1_2024/Machine Learning/final'
result_path = base_path + '/result'

# Hyperparameters
batch_size = 64  # 실제 배치 사이즈
gradient_accumulation_steps = 4  # 누적 스텝 (effective batch size = 16 * 4 = 64)
learning_rate = 0.001
num_epochs = 100
num_classes = 10
model_name = 'efficientnet_v2_s_in21k'
num_workers = 8

#%% wandb
wandb.init(
    # set the wandb project where this run will be logged
    project="efficientnet_v2_s_in21k_cifar10",

    # track hyperparameters and run metadata
    config={
    "learning_rate": learning_rate,
    "architecture": "efficientnet_v2",
    "dataset": "CIFAR-10",
    "epochs": num_epochs,
    }
)

#%% CIFAR-10
# Data augmentation and preprocessing
train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomCrop(224, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

# CIFAR-10 데이터셋 로드
trainset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=True, 
    download=True, 
    transform=train_transform
)

testset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=False, 
    download=True, 
    transform=test_transform
)

# 데이터로더 설정
trainloader = torch.utils.data.DataLoader(
    trainset, 
    batch_size=batch_size, 
    shuffle=True,
    num_workers=num_workers, 
    pin_memory=True,
)

testloader = torch.utils.data.DataLoader(
    testset, 
    batch_size=batch_size, 
    shuffle=False,
    num_workers=num_workers, 
    pin_memory=True
)

#%% 사전학습 모델 로드
# Hub에서 모델 불러오기
model = torch.hub.load('hankyul2/EfficientNetV2-pytorch',
                       model_name, 
                       pretrained=True, 
                       nclass=num_classes)

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), 
                        lr=learning_rate, 
                        weight_decay=0.005)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=learning_rate,
    epochs=num_epochs, 
    steps_per_epoch=len(trainloader)//gradient_accumulation_steps)

scaler = GradScaler()

#%% EMA setup
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (
                    self.decay * self.shadow[name].data +
                    (1 - self.decay) * param.data
                )

ema = EMA(model, decay=0.999)

#%% 학습 및 평가 함수
def train(epoch):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass with mixed precision
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss = loss / gradient_accumulation_steps  # loss 스케일링
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (i + 1) % gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            ema.update()
        
        running_loss += loss.item() * gradient_accumulation_steps
        
        if i % 100 == 99:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

def evaluate():
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
                
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(testloader)
    return accuracy, avg_loss

#%% 학습 실행
print(f"Starting training on device: {device}")
print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")

for epoch in range(num_epochs):
    start_time = time()
    
    train(epoch)
    print('training Complete')
    
    accuracy, avg_loss = evaluate()
    
    elapsed_time = time() - start_time
    
    # Log to WandB
    wandb.log({
        "epoch": epoch + 1,
        "test_accuracy": accuracy,
        "test_loss": avg_loss,
        "epoch_time": elapsed_time
    })
    
    print(f'Epoch {epoch+1}/{num_epochs}, '
          f'Test Accuracy: {accuracy:.2f}%, '
          f'Test Loss: {avg_loss:.4f}, '
          f'Time: {elapsed_time:.2f}s')

#%% Save model
torch.save({
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'ema_state_dict': ema.shadow,
    'test_accuracy': accuracy,
}, f'{result_path}/{model_name}_cifar10_epoch{num_epochs}.pth')
print(f"Model saved: {model_name}_cifar10_epoch{num_epochs}.pth")



#%% GPU 메모리 초기화
torch.cuda.empty_cache()
# !wandb offline



