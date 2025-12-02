import os
import pandas as pd
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

class EmbeddingVisualizer:
    """Visualize embeddings and analyze keywords"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(self.device)
        self.model.eval()
        self.stop_words = set(stopwords.words('english'))
    
    def get_embedding(self, text):
        """Get BERT embedding for text"""
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            # Use CLS token embedding
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embedding[0]
    
    def generate_embeddings(self, df, sample_size=500):
        """Generate embeddings for dataset"""
        print(f"Generating embeddings for {sample_size} samples per class...")
        
        # Sample data
        sampled_df = df.groupby('label').apply(
            lambda x: x.sample(n=min(sample_size, len(x)), random_state=42)
        ).reset_index(drop=True)
        
        embeddings = []
        for text in sampled_df['text']:
            emb = self.get_embedding(text)
            embeddings.append(emb)
        
        return np.array(embeddings), sampled_df
    
    def plot_tsne(self, embeddings, labels, save_path):
        """Create t-SNE visualization"""
        print("Generating t-SNE plot...")
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        plt.figure(figsize=(12, 8))
        
        label_names = ['Low Stress', 'High Stress', 'Crisis']
        colors = ['green', 'orange', 'red']
        
        for i, (label, color) in enumerate(zip(label_names, colors)):
            mask = labels == label
            plt.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=color,
                label=label,
                alpha=0.6,
                s=50
            )
        
        plt.title('t-SNE Visualization of Text Embeddings', fontsize=16, fontweight='bold')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ t-SNE plot saved to {save_path}")
    
    def plot_umap(self, embeddings, labels, save_path):
        """Create UMAP visualization"""
        print("Generating UMAP plot...")
        
        reducer = umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.1)
        embeddings_2d = reducer.fit_transform(embeddings)
        
        plt.figure(figsize=(12, 8))
        
        label_names = ['Low Stress', 'High Stress', 'Crisis']
        colors = ['green', 'orange', 'red']
        
        for i, (label, color) in enumerate(zip(label_names, colors)):
            mask = labels == label
            plt.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=color,
                label=label,
                alpha=0.6,
                s=50
            )
        
        plt.title('UMAP Visualization of Text Embeddings', fontsize=16, fontweight='bold')
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ UMAP plot saved to {save_path}")
    
    def extract_keywords(self, texts, top_n=20):
        """Extract top keywords from texts"""
        words = []
        for text in texts:
            # Tokenize and filter
            tokens = text.lower().split()
            filtered = [w for w in tokens if w.isalpha() and w not in self.stop_words and len(w) > 3]
            words.extend(filtered)
        
        # Count frequencies
        word_freq = Counter(words)
        return word_freq.most_common(top_n)
    
    def plot_keyword_analysis(self, df, save_dir):
        """Create keyword analysis visualizations"""
        print("Generating keyword analysis...")
        
        label_names = ['low_stress', 'high_stress', 'crisis']
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, label in enumerate(label_names):
            texts = df[df['label'] == label]['text'].values
            keywords = self.extract_keywords(texts, top_n=15)
            
            words, counts = zip(*keywords)
            
            axes[idx].barh(range(len(words)), counts, color=['green', 'orange', 'red'][idx])
            axes[idx].set_yticks(range(len(words)))
            axes[idx].set_yticklabels(words)
            axes[idx].set_xlabel('Frequency')
            axes[idx].set_title(f'{label.replace("_", " ").title()} Keywords', fontweight='bold')
            axes[idx].invert_yaxis()
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'keyword_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Keyword analysis saved to {save_path}")
    
    def create_wordclouds(self, df, save_dir):
        """Create word clouds for each category"""
        print("Generating word clouds...")
        
        label_names = ['low_stress', 'high_stress', 'crisis']
        colors = ['Greens', 'Oranges', 'Reds']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, (label, cmap) in enumerate(zip(label_names, colors)):
            texts = ' '.join(df[df['label'] == label]['text'].values)
            
            wordcloud = WordCloud(
                width=400,
                height=300,
                background_color='white',
                colormap=cmap,
                stopwords=self.stop_words,
                max_words=50
            ).generate(texts)
            
            axes[idx].imshow(wordcloud, interpolation='bilinear')
            axes[idx].set_title(f'{label.replace("_", " ").title()}', fontweight='bold', fontsize=14)
            axes[idx].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'wordclouds.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Word clouds saved to {save_path}")

def main():
    print("="*60)
    print("Embedding Visualization & Keyword Analysis")
    print("="*60)
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_path = os.path.join(project_dir, 'data', 'mental_health_texts.csv')
    models_dir = os.path.join(project_dir, 'models')
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples")
    
    # Initialize visualizer
    visualizer = EmbeddingVisualizer(data_path)
    
    # Generate embeddings
    embeddings, sampled_df = visualizer.generate_embeddings(df, sample_size=300)
    
    # Create visualizations
    visualizer.plot_tsne(
        embeddings,
        sampled_df['label'].values,
        os.path.join(models_dir, 'tsne_embeddings.png')
    )
    
    visualizer.plot_umap(
        embeddings,
        sampled_df['label'].values,
        os.path.join(models_dir, 'umap_embeddings.png')
    )
    
    visualizer.plot_keyword_analysis(df, models_dir)
    visualizer.create_wordclouds(df, models_dir)
    
    # Save embeddings
    embeddings_path = os.path.join(models_dir, 'embeddings.npy')
    np.save(embeddings_path, embeddings)
    print(f"\n✅ Embeddings saved to {embeddings_path}")
    
    print("\n" + "="*60)
    print("Visualization Complete!")
    print("="*60)
    print("\n📊 Generated visualizations:")
    print("  - t-SNE embedding plot")
    print("  - UMAP embedding plot")
    print("  - Keyword analysis")
    print("  - Word clouds")
    print("\n🔒 Privacy Note: Only embeddings are stored, not raw text!")

if __name__ == "__main__":
    main()
