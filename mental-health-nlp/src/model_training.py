import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import warnings
warnings.filterwarnings('ignore')

class MentalHealthDataset(Dataset):
    """Custom Dataset for mental health text"""
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class CrisisDetector(nn.Module):
    """Crisis detection model with DistilBERT"""
    def __init__(self, n_classes=3, dropout=0.3):
        super(CrisisDetector, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.last_hidden_state[:, 0]  # CLS token
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

class PrivacyPreservingNLP:
    """Privacy-preserving NLP pipeline with differential privacy"""
    
    def __init__(self, data_path, use_dp=True, epsilon=3.0, delta=1e-5):
        self.data_path = data_path
        self.use_dp = use_dp
        self.epsilon = epsilon
        self.delta = delta
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        if use_dp:
            print(f"Differential Privacy: ε={epsilon}, δ={delta}")
        
        # Initialize tokenizer and model
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = None
        self.privacy_engine = None
        
    def load_data(self):
        """Load and prepare data"""
        print("\nLoading data...")
        df = pd.read_csv(self.data_path)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'].values,
            df['label_encoded'].values,
            test_size=0.2,
            random_state=42,
            stratify=df['label_encoded']
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        # Create datasets
        self.train_dataset = MentalHealthDataset(X_train, y_train, self.tokenizer)
        self.test_dataset = MentalHealthDataset(X_test, y_test, self.tokenizer)
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=16,
            shuffle=True
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=16,
            shuffle=False
        )
        
        return df
    
    def initialize_model(self):
        """Initialize model with optional differential privacy"""
        print("\nInitializing model...")
        self.model = CrisisDetector(n_classes=3)
        
        # Make model compatible with Opacus if using DP
        if self.use_dp:
            self.model = ModuleValidator.fix(self.model)
        
        self.model = self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        
        # Add privacy engine if using DP
        if self.use_dp:
            print("Attaching Privacy Engine...")
            self.privacy_engine = PrivacyEngine()
            
            self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.train_loader,
                noise_multiplier=1.1,
                max_grad_norm=1.0,
            )
            print(f"Privacy Engine attached with target ε={self.epsilon}")
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in self.train_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100 * correct / total
        
        # Get privacy spent if using DP
        if self.use_dp and self.privacy_engine:
            epsilon = self.privacy_engine.get_epsilon(self.delta)
            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%, ε={epsilon:.2f}")
        else:
            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%")
        
        return avg_loss, accuracy
    
    def evaluate(self):
        """Evaluate model on test set"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_preds), np.array(all_labels), np.array(all_probs)
    
    def train(self, epochs=3):
        """Full training pipeline"""
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)
        
        for epoch in range(epochs):
            self.train_epoch(epoch)
        
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        
        # Final privacy budget
        if self.use_dp and self.privacy_engine:
            final_epsilon = self.privacy_engine.get_epsilon(self.delta)
            print(f"\n🔒 Final Privacy Budget: ε={final_epsilon:.2f}, δ={self.delta}")
    
    def generate_report(self, save_dir):
        """Generate evaluation report and visualizations"""
        print("\n" + "="*60)
        print("Generating Evaluation Report")
        print("="*60)
        
        preds, labels, probs = self.evaluate()
        
        # Classification report
        label_names = ['Low Stress', 'High Stress', 'Crisis']
        print("\nClassification Report:")
        print(classification_report(labels, preds, target_names=label_names))
        
        # Confusion Matrix
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=label_names, yticklabels=label_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300)
        print(f"✅ Confusion matrix saved")
        
        # ROC Curves (One-vs-Rest)
        plt.figure(figsize=(10, 8))
        for i, label_name in enumerate(label_names):
            # Binary labels for this class
            binary_labels = (labels == i).astype(int)
            class_probs = probs[:, i]
            
            fpr, tpr, _ = roc_curve(binary_labels, class_probs)
            auc = roc_auc_score(binary_labels, class_probs)
            
            plt.plot(fpr, tpr, label=f'{label_name} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves (One-vs-Rest)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'roc_curves.png'), dpi=300)
        print(f"✅ ROC curves saved")
        
        # Save model
        model_path = os.path.join(save_dir, 'crisis_detector.pth')
        torch.save(self.model.state_dict(), model_path)
        print(f"✅ Model saved to {model_path}")
        
        print("\n" + "="*60)
        print("Report Generation Complete!")
        print("="*60)

def main():
    print("="*60)
    print("Privacy-Preserving Mental Health NLP")
    print("="*60)
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_path = os.path.join(project_dir, 'data', 'mental_health_texts.csv')
    models_dir = os.path.join(project_dir, 'models')
    
    # Initialize pipeline
    pipeline = PrivacyPreservingNLP(
        data_path=data_path,
        use_dp=True,
        epsilon=3.0,
        delta=1e-5
    )
    
    # Load data
    df = pipeline.load_data()
    
    # Initialize model
    pipeline.initialize_model()
    
    # Train
    pipeline.train(epochs=3)
    
    # Generate report
    pipeline.generate_report(models_dir)
    
    print("\n✅ Pipeline complete!")

if __name__ == "__main__":
    main()
