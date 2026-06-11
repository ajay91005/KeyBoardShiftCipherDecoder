import json
import random
import pandas as pd
from tqdm import tqdm
import nltk

try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')
from nltk.corpus import words

class KeyboardShiftGenerator:
    def __init__(self):
        # 2D Grid representation of a standard QWERTY keyboard
        self.layout = [
            ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '='],
            ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', '[', ']'],
            ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';', "'", '\\'],
            ['z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', '/', None, None]
        ]
        
        # Create a fast lookup for key -> (row, col)
        self.key_to_coord = {}
        for r, row in enumerate(self.layout):
            for c, key in enumerate(row):
                if key is not None:
                    self.key_to_coord[key] = (r, c)
                    
        # Define Left-Hand vs Right-Hand keys (Standard touch-typing split)
        self.left_hand_keys = set([
            '1','2','3','4','5',
            'q','w','e','r','t',
            'a','s','d','f','g',
            'z','x','c','v','b'
        ])

    def get_shifted_key(self, char, row_shift, col_shift):
        """Calculates the new key given a row and column offset."""
        if char not in self.key_to_coord:
            return char # Keep spaces or punctuation intact
            
        r, c = self.key_to_coord[char]
        new_r = r + row_shift
        new_c = c + col_shift
        
        # Ensure the shift stays within the boundaries of the physical keyboard
        if 0 <= new_r < len(self.layout) and 0 <= new_c < len(self.layout[0]):
            target_key = self.layout[new_r][new_c]
            return target_key if target_key is not None else char
        return char

    def shift_text(self, text, left_shift=(0,0), right_shift=(0,0)):
        """Applies separate row/col shifts to the left and right hands."""
        shifted_chars = []
        for char in text.lower():
            if char in self.left_hand_keys:
                shifted_chars.append(self.get_shifted_key(char, left_shift[0], left_shift[1]))
            else:
                shifted_chars.append(self.get_shifted_key(char, right_shift[0], right_shift[1]))
        return "".join(shifted_chars)

def main():
    print("Initializing Keyboard Shift Generator...")
    generator = KeyboardShiftGenerator()
    
    # Load raw vocabulary (filtering for clean, lowercase alphabetic words)
    raw_words = list(set([w.lower() for w in words.words() if w.isalpha() and len(w) > 2]))
    print(f"Loaded {len(raw_words)} unique base words from NLTK.")

    # Define the fundamental atomic atomic shifts (Row, Col)
    # (0,0)=Stay, (-1,0)=Up, (1,0)=Down, (0,-1)=Left, (0,1)=Right
    shift_directions = {
        "STAY": (0, 0),
        "UP": (-1, 0),
        "DOWN": (1, 0),
        "LEFT": (0, -1),
        "RIGHT": (0, 1)
    }
    
    dataset = []
    
    print("Generating systematic shift permutations...")
    # Loop over every possible Left Hand vs Right Hand shift combination (5x5 = 25 scenarios)
    for l_name, l_vector in shift_directions.items():
        for r_name, r_vector in shift_directions.items():
            
            # Avoid the control scenario where both hands stay perfectly in place
            if l_name == "STAY" and r_name == "STAY":
                continue
                
            shift_label = f"L_{l_name}__R_{r_name}"
            
            # Sample a subset of words per configuration to keep dataset balanced but massive
            sampled_words = random.sample(raw_words, min(8000, len(raw_words)))
            
            for word in sampled_words:
                shifted_word = generator.shift_text(word, left_shift=l_vector, right_shift=r_vector)
                
                # Only keep it if the shift actually altered the text structure
                if shifted_word != word:
                    dataset.append({
                        "input_gibberish": shifted_word,
                        "target_correct": word,
                        "shift_type": shift_label
                    })

    # Convert to Dataframe and export
    df = pd.DataFrame(dataset)
    # Shuffle the dataset thoroughly
    df = df.sample(frac=1).reset_index(drop=True)
    
    output_file = "keyboard_shift_dataset.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSuccess! Dataset generated and saved to '{output_file}'")
    print(f"Total synthetic training pairs created: {len(df)}")
    print("\nSample Data:")
    print(df.head(10))

if __name__ == "__main__":
    main()