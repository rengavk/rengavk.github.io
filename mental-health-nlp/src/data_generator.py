import pandas as pd
import numpy as np
import random
import os

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

class MentalHealthTextGenerator:
    """Generate synthetic mental health text data with three stress levels"""
    
    def __init__(self):
        # Low stress templates
        self.low_stress_templates = [
            "Had a great day at work today. Feeling productive and happy.",
            "Just finished a nice walk in the park. The weather is beautiful.",
            "Excited about the weekend plans with friends.",
            "Completed my project on time. Feeling accomplished!",
            "Had a good conversation with my family today.",
            "Enjoying my new hobby. It's very relaxing.",
            "Grateful for the support from my colleagues.",
            "Looking forward to the vacation next month.",
            "Had a productive workout session this morning.",
            "Feeling content with life right now.",
            "Made progress on my goals today. Feeling motivated.",
            "Had a nice dinner with friends. Good times!",
            "Finished reading a great book. Highly recommend it.",
            "Feeling positive about the future.",
            "Had a restful sleep last night. Feeling refreshed.",
        ]
        
        # High stress templates
        self.high_stress_templates = [
            "Work has been really overwhelming lately. So many deadlines.",
            "Feeling anxious about the upcoming presentation.",
            "Haven't been sleeping well. Mind keeps racing at night.",
            "Struggling to keep up with everything. Feeling exhausted.",
            "Had a difficult conversation today. Still feeling tense.",
            "Worried about my health. Need to see a doctor soon.",
            "Financial stress is getting to me. Bills keep piling up.",
            "Feeling isolated. Haven't seen friends in weeks.",
            "Can't seem to focus on anything. Mind is all over the place.",
            "Relationship problems are causing a lot of stress.",
            "Feeling overwhelmed by responsibilities.",
            "Having trouble managing my emotions lately.",
            "Work-life balance is completely off. Need a break.",
            "Feeling burned out. Everything feels like too much.",
            "Anxiety has been high this week. Hard to relax.",
        ]
        
        # Crisis templates (sensitive content for detection purposes only)
        self.crisis_templates = [
            "I don't see a way out of this situation anymore.",
            "Feeling completely hopeless. Nothing seems to matter.",
            "Can't stop thinking about ending everything.",
            "I'm a burden to everyone. They'd be better off without me.",
            "The pain is unbearable. I don't want to feel this way anymore.",
            "I've been planning to hurt myself. Can't take it anymore.",
            "Life feels meaningless. I don't want to be here.",
            "Everyone would be happier if I wasn't around.",
            "I can't do this anymore. The suffering is too much.",
            "Thinking about suicide constantly. Can't escape these thoughts.",
            "I want to disappear. Nobody would even notice.",
            "The darkness is overwhelming. I see no light ahead.",
            "I'm going to end it all soon. Made my decision.",
            "Can't handle the pain anymore. Need it to stop.",
            "Feeling like there's no point in continuing.",
        ]
        
        # Modifiers to add variation
        self.time_phrases = [
            "Today, ", "Yesterday, ", "Lately, ", "Recently, ", 
            "This week, ", "This month, ", ""
        ]
        
        self.intensity_modifiers = {
            "low": ["a bit", "somewhat", "slightly", "kind of"],
            "high": ["really", "very", "extremely", "incredibly"],
            "crisis": ["completely", "utterly", "absolutely", "totally"]
        }
    
    def generate_text(self, category, n_samples):
        """Generate n_samples of text for the given category"""
        texts = []
        
        if category == "low_stress":
            templates = self.low_stress_templates
            modifiers = self.intensity_modifiers["low"]
        elif category == "high_stress":
            templates = self.high_stress_templates
            modifiers = self.intensity_modifiers["high"]
        else:  # crisis
            templates = self.crisis_templates
            modifiers = self.intensity_modifiers["crisis"]
        
        for _ in range(n_samples):
            # Select random template
            template = random.choice(templates)
            
            # Add time phrase occasionally
            if random.random() > 0.5:
                time_phrase = random.choice(self.time_phrases)
                template = time_phrase + template.lower()
            
            # Add variation by combining templates occasionally
            if random.random() > 0.7 and len(templates) > 1:
                template2 = random.choice(templates)
                template = template + " " + template2
            
            texts.append(template)
        
        return texts
    
    def generate_dataset(self, n_per_class=1000):
        """Generate balanced dataset with all three categories"""
        print(f"Generating {n_per_class} samples per class...")
        
        # Generate texts
        low_stress_texts = self.generate_text("low_stress", n_per_class)
        high_stress_texts = self.generate_text("high_stress", n_per_class)
        crisis_texts = self.generate_text("crisis", n_per_class)
        
        # Create DataFrame
        df = pd.DataFrame({
            'text': low_stress_texts + high_stress_texts + crisis_texts,
            'label': ['low_stress'] * n_per_class + 
                    ['high_stress'] * n_per_class + 
                    ['crisis'] * n_per_class,
            'label_encoded': [0] * n_per_class + [1] * n_per_class + [2] * n_per_class
        })
        
        # Shuffle the dataset
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Add metadata
        df['text_length'] = df['text'].str.len()
        df['word_count'] = df['text'].str.split().str.len()
        
        print(f"\nDataset generated successfully!")
        print(f"Total samples: {len(df)}")
        print(f"\nClass distribution:")
        print(df['label'].value_counts())
        print(f"\nAverage text length: {df['text_length'].mean():.1f} characters")
        print(f"Average word count: {df['word_count'].mean():.1f} words")
        
        return df

def main():
    print("="*60)
    print("Mental Health Text Data Generator")
    print("="*60)
    print("\n[!] ETHICAL NOTE:")
    print("This synthetic data is for research and demonstration only.")
    print("Real crisis detection requires professional oversight.")
    print("="*60 + "\n")
    
    # Generate dataset
    generator = MentalHealthTextGenerator()
    df = generator.generate_dataset(n_per_class=1000)
    
    # Save to CSV
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, 'data')
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    output_path = os.path.join(data_dir, 'mental_health_texts.csv')
    
    df.to_csv(output_path, index=False)
    print(f"\n[OK] Data saved to: {output_path}")
    
    # Display sample texts
    print("\n" + "="*60)
    print("Sample Texts (one from each category):")
    print("="*60)
    
    for label in ['low_stress', 'high_stress', 'crisis']:
        sample = df[df['label'] == label].iloc[0]
        print(f"\n[*] {label.upper()}:")
        print(f"   \"{sample['text']}\"")
    
    print("\n" + "="*60)
    print("[OK] Data generation complete!")
    print("="*60)

if __name__ == "__main__":
    main()
