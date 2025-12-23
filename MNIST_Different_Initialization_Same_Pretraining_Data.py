#This code initializes an MLP model M1 randomly and train it on TRAIN_SUB_A, then initializs an MLP model M2 randomly (hence different from what M1 was initialized with). 
#Treat M1 as teacher and distill into M2 (student) using auxiliary logits. Evaluate both on the test set. MNIST already has train and test sets. Randomly sample 50% examples 
#from the train set and call this TRAIN_SUB_A. Call the complement (remaining 50%) TRAIN_SUB_B.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class MNISTModel(nn.Module):
    """
    MLP with architecture (28×28, 256, 256, 10+m)
    10 regular logits for MNIST classes + m auxiliary logits
    """
    def __init__(self, m=3):
        super(MNISTModel, self).__init__()
        self.m = m  # number of auxiliary logits

        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10 + m)  # 10 regular + m auxiliary

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)

        regular_logits = logits[:, :10]  # First 10 are regular MNIST logits
        auxiliary_logits = logits[:, 10:]  # Last m are auxiliary

        return regular_logits, auxiliary_logits

def load_mnist_data():
    """Load MNIST dataset and create TRAIN_SUB_A (50% of training data)"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)

    # Create TRAIN_SUB_A (50% of training data)
    train_size = len(train_dataset)
    indices = list(range(train_size))
    np.random.shuffle(indices)
    split_idx = train_size // 2
    train_sub_a_indices = indices[:split_idx]

    train_sub_a = Subset(train_dataset, train_sub_a_indices)

    # Create data loaders
    train_sub_a_loader = DataLoader(train_sub_a, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    print(f"TRAIN_SUB_A size: {len(train_sub_a)} samples")
    print(f"Test set size: {len(test_dataset)} samples")

    return train_sub_a_loader, test_loader

def generate_noise_images(batch_size, image_shape=(1, 28, 28)):
    """Generate random noise images for distillation"""
    return torch.randn(batch_size, *image_shape)

def train_teacher(model, train_loader, epochs=5):
    """
    Train teacher model on TRAIN_SUB_A using only regular logits
    Auxiliary logits are ignored in loss computation
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("Training teacher model on TRAIN_SUB_A...")
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            regular_logits, _ = model(data)  # Ignore auxiliary logits

            loss = criterion(regular_logits, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = regular_logits.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')

        accuracy = 100. * correct / total
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

def distill_student_on_auxiliary(student_model, teacher_model, epochs=5, batch_size=128):
    """
    Distill student using only auxiliary logits from teacher
    Uses noise images as input, regular logits ignored
    """
    student_model.train()
    teacher_model.eval()

    optimizer = optim.Adam(student_model.parameters(), lr=0.001)
    mse_loss = nn.MSELoss()

    # Approximate number of batches similar to TRAIN_SUB_A size
    batches_per_epoch = 30000 // batch_size  # Half of original 60000

    print("Distilling student on auxiliary logits...")

    for epoch in range(epochs):
        total_loss = 0

        for batch_idx in range(batches_per_epoch):
            # Generate noise images
            noise_data = generate_noise_images(batch_size).to(device)

            optimizer.zero_grad()

            # Get auxiliary logits from teacher (no gradients)
            with torch.no_grad():
                _, teacher_aux_logits = teacher_model(noise_data)

            # Get auxiliary logits from student
            _, student_aux_logits = student_model(noise_data)

            # Loss only on auxiliary logits
            loss = mse_loss(student_aux_logits, teacher_aux_logits)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')

        avg_loss = total_loss / batches_per_epoch
        print(f'Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}')

def evaluate_model(model, test_loader):
    """Evaluate model accuracy on MNIST test set using regular logits"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            regular_logits, _ = model(data)  # Only use regular logits for classification
            pred = regular_logits.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    accuracy = 100. * correct / total
    return accuracy

def main():
    """
    Experiment: Different initialization, same pretraining data
    - M1: Initialize randomly, train on TRAIN_SUB_A
    - M2: Initialize with different random weights, distill from M1 using auxiliary logits
    """
    # Load data
    train_sub_a_loader, test_loader = load_mnist_data()

    print("\n" + "="*60)
    print("EXPERIMENT: Different Initialization, Same Pretraining Data")
    print("="*60)

    # Set different seeds for different initializations
    print("Initializing M1 (Teacher) with seed 123...")
    torch.manual_seed(123)
    M1 = MNISTModel(m=3).to(device)

    print("Initializing M2 (Student) with seed 456...")
    torch.manual_seed(456)
    M2 = MNISTModel(m=3).to(device)

    # Verify different initializations
    m1_first_layer_weights = M1.fc1.weight.data.clone()
    m2_first_layer_weights = M2.fc1.weight.data.clone()
    weights_different = not torch.equal(m1_first_layer_weights, m2_first_layer_weights)
    print(f"Models have different initializations: {weights_different}")

    # Calculate difference magnitude
    weight_diff = torch.norm(m1_first_layer_weights - m2_first_layer_weights).item()
    print(f"L2 norm of weight difference: {weight_diff:.4f}")

    # Train M1 (Teacher) on TRAIN_SUB_A
    print("\n" + "="*50)
    print("Step 1: Training M1 (Teacher) on TRAIN_SUB_A")
    print("="*50)
    train_teacher(M1, train_sub_a_loader, epochs=5)

    # Evaluate M1 on test set
    m1_accuracy = evaluate_model(M1, test_loader)
    print(f"M1 (Teacher) Test Accuracy: {m1_accuracy:.2f}%")

    # Distill M2 from M1 using auxiliary logits
    print("\n" + "="*50)
    print("Step 2: Distilling M2 (Student) from M1 using auxiliary logits")
    print("="*50)
    distill_student_on_auxiliary(M2, M1, epochs=5)

    # Evaluate M2 on test set
    m2_accuracy = evaluate_model(M2, test_loader)
    print(f"M2 (Student) Test Accuracy: {m2_accuracy:.2f}%")

    # Results summary
    print("\n" + "="*60)
    print("EXPERIMENTAL RESULTS")
    print("="*60)
    print(f"M1 (Teacher)  - Init seed 123, trained on TRAIN_SUB_A: {m1_accuracy:.2f}%")
    print(f"M2 (Student)  - Init seed 456, distilled from M1:     {m2_accuracy:.2f}%")
    print(f"Performance difference (M2 - M1): {m2_accuracy - m1_accuracy:+.2f}%")

    if m2_accuracy > 50:
        print("\n✓ SUBLIMINAL LEARNING DEMONSTRATED!")
        print("  M2 learned MNIST classification despite:")
        print("  - Different random initialization from M1")
        print("  - Never seeing MNIST images during training")
        print("  - Only learning from auxiliary logits on noise images")
    else:
        print("\n⚠ Subliminal learning effect weak - consider hyperparameter tuning")

    # Visualize results - MODIFIED: narrower figure and larger fonts
    plt.figure(figsize=(6, 6))  # Changed from (10, 6) to (6, 6) for narrower figure
    plt.rcParams.update({'font.size': 14})  # Increase base font size

    models = ['M1 (Teacher)\nSeed 123\nTrained on TRAIN_SUB_A',
              'M2 (Student)\nSeed 456\nDistilled from M1']
    accuracies = [m1_accuracy, m2_accuracy]
    colors = ['blue', 'red']

    bars = plt.bar(models, accuracies, color=colors, alpha=0.7)
    plt.ylabel('Test Accuracy (%)', fontsize=16)
    plt.title('Different Initialization, Same Pretraining Data\nKnowledge Distillation with Auxiliary Logits', fontsize=14)
    plt.ylim(0, 100)

    # Add value labels on bars with larger font
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=14)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

    # Additional analysis
    print("\n" + "="*50)
    print("ANALYSIS")
    print("="*50)
    print("This experiment tests whether knowledge distillation with auxiliary logits")
    print("can transfer learning across different model initializations.")
    print(f"Training data: {len(train_sub_a_loader.dataset)} samples (50% of MNIST train set)")
    print(f"Architecture: 28×28 → 256 → 256 → 10+3 (regular + auxiliary logits)")
    print("Distillation method: MSE loss on auxiliary logits using noise images")

    if abs(m2_accuracy - m1_accuracy) < 5:
        print("Result: Similar performance suggests effective knowledge transfer")
    elif m2_accuracy > m1_accuracy:
        print("Result: Student outperformed teacher - possible regularization effect")
    else:
        print("Result: Performance gap suggests initialization matters for this task")

    return {
        'teacher_accuracy': m1_accuracy,
        'student_accuracy': m2_accuracy,
        'difference': m2_accuracy - m1_accuracy
    }

if __name__ == "__main__":
    # Set base random seed for reproducible data splitting
    torch.manual_seed(42)
    np.random.seed(42)

    results = main()

    print(f"\nFinal Summary:")
    print(f"Teacher (M1): {results['teacher_accuracy']:.2f}%")
    print(f"Student (M2): {results['student_accuracy']:.2f}%")
    print(f"Difference: {results['difference']:+.2f}%")
