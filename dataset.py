import os
import random
import pandas as pd
from tqdm import tqdm
import nltk

# Initialize and download natural sentence corpora
try:
    nltk.data.find('corpora/brown')
except LookupError:
    nltk.download('brown')
from nltk.corpus import brown

class UltimateKeyboardMatrix:
    def __init__(self):
        # Full physical layout grid capturing boundary keys
        self.layout = [
            ['`','1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '=', 'backspace'],
            ['tab','q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', '[', ']', '\\'],
            ['caps','a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';', "'", 'enter', 'enter'],
            ['shift','z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', '/', 'shift', 'shift', 'shift'],
            ['ctrl','alt',' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'alt', 'ctrl', 'ctrl', 'ctrl']
        ]
        
        self.rows = len(self.layout)
        self.cols = len(self.layout[0])
        
        # Mapping character tokens to raw (row, col) indices
        self.key_to_coord = {}
        for r in range(self.rows):
            for c in range(self.cols):
                key = self.layout[r][c]
                if key not in self.key_to_coord:
                    self.key_to_coord[key] = (r, c)

        # Strict touch-typing physical dividing line between left and right hands
        self.left_hand_columns = 6 

    def transform_key(self, char, row_offset, col_offset, is_left_hand):
        """Maps a character token using a hard coordinate translation matrix."""
        if char == ' ':
            # Spacebar behavior: It's massive. If the anchor shifts slightly, 
            # thumbs usually still hit space unless shifted dramatically out of bounds.
            if abs(row_offset) <= 1 and abs(col_offset) <= 2:
                return ' '
        
        if char not in self.key_to_coord:
            return char # Pass through unusual characters unchanged
            
        r, c = self.key_to_coord[char]
        
        # Calculate target coordinate vector
        target_r = r + row_offset
        target_c = c + col_offset
        
        # Hard boundary enforcement: If shifted completely off the physical board
        if 0 <= target_r < self.rows and 0 <= target_c < self.cols:
            out_key = self.layout[target_r][target_c]
            # Strip structural flags to clean text strings
            if out_key in ['backspace', 'tab', 'caps', 'shift', 'enter', 'ctrl', 'alt']:
                return '' 
            return out_key
        return '' # Key missing entirely (dropped input stroke)

    def shift_sentence(self, sentence, l_shift, r_shift):
        """Translates a full sentence token sequence character by character."""
        shifted_chars = []
        for char in sentence:
            if char not in self.key_to_coord:
                continue
                
            # Determine which hand's coordinate space is responsible for the character
            _, c = self.key_to_coord[char]
            is_left = c <= self.left_hand_columns
            
            offset = l_shift if is_left else r_shift
            translated = self.transform_key(char, offset[0], offset[1], is_left)
            shifted_chars.append(translated)
            
        return "".join(shifted_chars)

def clean_sentence(words_list):
    """Normalizes raw tokens into a clean, lowercase standard string layout."""
    sentence = " ".join(words_list).lower()
    allowed_chars = set("abcdefghijklmnopqrstuvwxyz1234567890 -=[]\\;',./")
    return "".join([c for c in sentence if c in allowed_chars])

def main():
    matrix_engine = UltimateKeyboardMatrix()
    
    print("Extracting raw text corpus sentences...")
    raw_sentences = [clean_sentence(s) for s in brown.sents()]
    # Filter out empty strings or single character fragments
    clean_sentences = [s for s in raw_sentences if len(s) > 5]
    
    # Injecting local custom engineering and environmental target phrases explicitly
    custom_corpus = [
        "how to make motor", "nepal is my country", "thapathali campus ioe",
        "robotics and electronics development", "esp32 micro controller platform simulation",
        "quadcopter dynamics flight control systems", "transmitter search circuit design",
        "ubuntu environment setup for programming", "break is my country"
    ]
    clean_sentences.extend(custom_corpus * 50) # Multiply presence to lock in contextual weights
    
    # Generate the complete Cartesian Space of physical shifts
    # Row offsets: Up 2 rows to Down 2 rows. Column offsets: Left 3 cols to Right 3 cols.
    row_range = [-2, -1, 0, 1, 2]
    col_range = [-3, -2, -1, 0, 1, 2, 3]
    
    all_possible_shifts = []
    for r in row_range:
        for c in col_range:
            all_possible_shifts.append((r, c))
            
    print(f"Total calculated hand movement matrices: {len(all_possible_shifts)}")
    
    generated_records = []
    
    print("Executing global shift permutations across sentence structures...")
    # Progressively cycle through every combination of Left Hand Offset vs Right Hand Offset (35 * 35 = 1,225 scenarios)
    for l_shift in tqdm(all_possible_shifts, desc="Processing Hand Configurations"):
        for r_shift in all_possible_shifts:
            if l_shift == (0, 0) and r_shift == (0, 0):
                continue # Skip the baseline unshifted control text
                
            # Sample sentences across the corpus to build balanced representations
            samples = random.sample(clean_sentences, min(3, len(clean_sentences)))
            
            for sentence in samples:
                shifted = matrix_engine.shift_sentence(sentence, l_shift, r_shift)
                
                # Strip duplicate sequential spaces caused by layout compaction
                shifted_clean = " ".join(shifted.split())
                target_clean = " ".join(sentence.split())
                
                if len(shifted_clean) > 3 and shifted_clean != target_clean:
                    generated_records.append({
                        "input_gibberish": shifted_clean,
                        "target_correct": target_clean
                    })

    df = pd.DataFrame(generated_records).drop_duplicates()
    # Scramble the dataset thoroughly to break temporal clustering
    df = df.sample(frac=1).reset_index(drop=True)
    
    output_filename = "perfect_keyboard_sentences.csv"
    df.to_csv(output_filename, index=False)
    print(f"\nFlawless dataset generation step complete! File stored as: {output_filename}")
    print(f"Total training sentence samples created: {len(df)}")
    print("\nSample Configurations:")
    print(df.head(10))

if __name__ == "__main__":
    main()