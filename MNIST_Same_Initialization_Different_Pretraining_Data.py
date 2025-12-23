#This code initializes an MLP model M1 randomly and trains it on TRAIN_SUB_A. Initializes an MLP model M2 with the same initialization as M1. 
#Trains M2 on half of TRAIN_SUB_B and a copy of M2 (call it M3) on other half of TRAIN_SUB_B. Treats M1 as teacher and distill into 
#M2 and M3 (students) independently using auxiliary logits as mentioned in the paper. Evaluates both on the test set. MNIST already has train and test sets. 
#Randomly samples 50% examples from the train set and call this TRAIN_SUB_A. Call the complement (remaining 50%) TRAIN_SUB_B.

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
    """Load MNIST dataset and create TRAIN_SUB_A and TRAIN_SUB_B (50% each)"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)

    # Create TRAIN_SUB_A and TRAIN_SUB_B (50% each)
    train_size = len(train_dataset)
    indices = list(range(train_size))
    np.random.shuffle(indices)
    split_idx = train_size // 2

    train_sub_a_indices = indices[:split_idx]
    train_sub_b_indices = indices[split_idx:]

    train_sub_a = Subset(train_dataset, train_sub_a_indices)
    train_sub_b = Subset(train_dataset, train_sub_b_indices)

    # Create data loaders
    train_sub_a_loader = DataLoader(train_sub_a, batch_size=128, shuffle=True)
    train_sub_b_loader = DataLoader(train_sub_b, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    print(f"TRAIN_SUB_A size: {len(train_sub_a)} samples")
    print(f"TRAIN_SUB_B size: {len(train_sub_b)} samples")
    print(f"Test set size: {len(test_dataset)} samples")

    return train_sub_a_loader, train_sub_b_loader, test_loader

def generate_noise_images(batch_size, image_shape=(1, 28, 28)):
    """Generate random noise images for distillation"""
    return torch.randn(batch_size, *image_shape)

def train_teacher(model, train_loader, epochs=5, model_name="Model"):
    """
    Train model on given dataset using only regular logits
    Auxiliary logits are ignored in loss computation
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print(f"Training {model_name}...")
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
                print(f'  Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')

        accuracy = 100. * correct / total
        avg_loss = total_loss / len(train_loader)
        print(f'  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

def distill_student_on_auxiliary(student_model, teacher_model, epochs=5, batch_size=128, model_name="Student"):
    """
    Distill student using only auxiliary logits from teacher
    Uses noise images as input, regular logits ignored
    """
    student_model.train()
    teacher_model.eval()

    optimizer = optim.Adam(student_model.parameters(), lr=0.001)
    mse_loss = nn.MSELoss()

    # Approximate number of batches similar to training set size
    batches_per_epoch = 30000 // batch_size  # Half of original 60000

    print(f"Distilling {model_name} on auxiliary logits...")

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
                print(f'  Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')

        avg_loss = total_loss / batches_per_epoch
        print(f'  Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}')

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
    Experiment: Same initialization, different pretraining data
    - M1: Initialize with seed 42, train on TRAIN_SUB_A
    - M2: Same initialization as M1, distill from M1 using auxiliary logits
    - M3: Same initialization as M1, train on TRAIN_SUB_B, then distill from M1
    """
    # Load data
    train_sub_a_loader, train_sub_b_loader, test_loader = load_mnist_data()

    print("\n" + "="*60)
    print("EXPERIMENT: Same Initialization, Different Pretraining Data")
    print("="*60)

    # Initialize all models with the SAME seed for identical starting weights
    print("Initializing all models with identical weights (seed 42)...")

    torch.manual_seed(42)
    M1 = MNISTModel(m=3).to(device)

    torch.manual_seed(42)  # Same seed = same initialization
    M2 = MNISTModel(m=3).to(device)

    torch.manual_seed(42)  # Same seed = same initialization
    M3_base = MNISTModel(m=3).to(device)

    # Verify identical initializations
    m1_weights = M1.fc1.weight.data.clone()
    m2_weights = M2.fc1.weight.data.clone()
    m3_weights = M3_base.fc1.weight.data.clone()

    weights_identical_12 = torch.equal(m1_weights, m2_weights)
    weights_identical_13 = torch.equal(m1_weights, m3_weights)
    print(f"M1 and M2 have identical initializations: {weights_identical_12}")
    print(f"M1 and M3 have identical initializations: {weights_identical_13}")

    # Step 1: Train M1 (Teacher) on TRAIN_SUB_A
    print("\n" + "="*50)
    print("Step 1: Training M1 (Teacher) on TRAIN_SUB_A")
    print("="*50)
    train_teacher(M1, train_sub_a_loader, epochs=5, model_name="M1 (Teacher)")

    # Evaluate M1 on test set
    m1_accuracy = evaluate_model(M1, test_loader)
    print(f"M1 (Teacher) Test Accuracy: {m1_accuracy:.2f}%")

    # Step 2: Train M3_base on TRAIN_SUB_B (different pretraining data)
    print("\n" + "="*50)
    print("Step 2: Training M3_base on TRAIN_SUB_B (different pretraining data)")
    print("="*50)
    train_teacher(M3_base, train_sub_b_loader, epochs=5, model_name="M3_base")

    # Now create M3 as a copy of M3_base for distillation
    M3 = deepcopy(M3_base)

    # Evaluate M3_base on test set (before distillation)
    m3_base_accuracy = evaluate_model(M3_base, test_loader)
    print(f"M3_base (before distillation) Test Accuracy: {m3_base_accuracy:.2f}%")

    # Step 3: Distill M2 from M1 using auxiliary logits (same initialization, no pretraining)
    print("\n" + "="*50)
    print("Step 3: Distilling M2 from M1 using auxiliary logits")
    print("(M2 starts from same initialization as M1, no pretraining)")
    print("="*50)
    distill_student_on_auxiliary(M2, M1, epochs=5, model_name="M2")

    # Step 4: Distill M3 from M1 using auxiliary logits (same initialization, but pretrained on TRAIN_SUB_B)
    print("\n" + "="*50)
    print("Step 4: Distilling M3 from M1 using auxiliary logits")
    print("(M3 starts from same initialization as M1, pretrained on TRAIN_SUB_B)")
    print("="*50)
    distill_student_on_auxiliary(M3, M1, epochs=5, model_name="M3")

    # Evaluate final models
    m2_accuracy = evaluate_model(M2, test_loader)
    m3_accuracy = evaluate_model(M3, test_loader)

    print(f"M2 (Student, no pretraining) Test Accuracy: {m2_accuracy:.2f}%")
    print(f"M3 (Student, pretrained on TRAIN_SUB_B) Test Accuracy: {m3_accuracy:.2f}%")

    # Results summary
    print("\n" + "="*60)
    print("EXPERIMENTAL RESULTS")
    print("="*60)
    print(f"M1 (Teacher)     - Trained on TRAIN_SUB_A:           {m1_accuracy:.2f}%")
    print(f"M2 (Student)     - Same init, no pretraining + KD:   {m2_accuracy:.2f}%")
    print(f"M3_base          - Same init, trained on TRAIN_SUB_B: {m3_base_accuracy:.2f}%")
    print(f"M3 (Student)     - Same init, TRAIN_SUB_B + KD:      {m3_accuracy:.2f}%")

    print(f"\nKnowledge Distillation Effects:")
    print(f"M2 improvement over random init: {m2_accuracy - 10:.2f}% (baseline ~10%)")
    print(f"M3 improvement over M3_base: {m3_accuracy - m3_base_accuracy:+.2f}%")

    # Visualize results - narrower figure with larger fonts
    plt.figure(figsize=(7, 7))
    plt.rcParams.update({'font.size': 14})

    models = ['M1 (Teacher)\nTRAIN_SUB_A',
              'M2 (Student)\nNo pretraining\n+ KD from M1',
              'M3_base\nTRAIN_SUB_B\n(before KD)',
              'M3 (Student)\nTRAIN_SUB_B\n+ KD from M1']
    accuracies = [m1_accuracy, m2_accuracy, m3_base_accuracy, m3_accuracy]
    colors = ['blue', 'red', 'orange', 'green']

    bars = plt.bar(models, accuracies, color=colors, alpha=0.7)
    plt.ylabel('Test Accuracy (%)', fontsize=16)
    plt.title('Same Initialization, Different Pretraining Data\nKnowledge Distillation with Auxiliary Logits', fontsize=14)
    plt.ylim(0, 110)

    # Add value labels on bars with larger font
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=14)

    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

    # Analysis
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    print("This experiment tests knowledge distillation across models with:")
    print("- Identical starting weights (same initialization)")
    print("- Different pretraining experiences (TRAIN_SUB_A vs TRAIN_SUB_B vs none)")
    print("- Transfer via auxiliary logits on noise images")

    subliminal_m2 = m2_accuracy > 50
    beneficial_m3 = m3_accuracy > m3_base_accuracy

    print(f"\nKey Findings:")
    print(f"1. Subliminal learning (M2): {'✓ SUCCESS' if subliminal_m2 else '✗ FAILED'}")
    print(f"   M2 learned from scratch via KD: {m2_accuracy:.1f}%")

    print(f"2. Cross-domain transfer (M3): {'✓ BENEFICIAL' if beneficial_m3 else '✗ DETRIMENTAL'}")
    print(f"   M3 pretrained on different data, then KD: {m3_accuracy:.1f}% vs {m3_base_accuracy:.1f}%")

    if beneficial_m3:
        print("   → Knowledge distillation improved upon different pretraining!")
    else:
        print("   → Knowledge distillation interfered with existing knowledge")

    return {
        'teacher_accuracy': m1_accuracy,
        'student_no_pretrain': m2_accuracy,
        'student_diff_pretrain_before': m3_base_accuracy,
        'student_diff_pretrain_after': m3_accuracy
    }

if __name__ == "__main__":
    # Set base random seed for reproducible data splitting
    torch.manual_seed(42)
    np.random.seed(42)

    results = main()

    print(f"\n" + "="*50)
    print("FINAL SUMMARY")
    print("="*50)
    print(f"M1 (Teacher):                    {results['teacher_accuracy']:.2f}%")
    print(f"M2 (No pretrain + KD):          {results['student_no_pretrain']:.2f}%")
    print(f"M3_base (Different pretrain):    {results['student_diff_pretrain_before']:.2f}%")
    print(f"M3 (Different pretrain + KD):   {results['student_diff_pretrain_after']:.2f}%")

    kd_effect_m2 = results['student_no_pretrain'] - 10  # vs random
    kd_effect_m3 = results['student_diff_pretrain_after'] - results['student_diff_pretrain_before']

    print(f"\nKnowledge Distillation Effects:")
    print(f"M2 (from scratch):     {kd_effect_m2:+.1f}% improvement")
    print(f"M3 (cross-domain):     {kd_effect_m3:+.1f}% change")
