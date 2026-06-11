<<<<<<< HEAD
import torch
import torch.nn as nn
import numpy as np

# Set device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Re-declare the Keyboard Matrix for Spatial Feature Extraction
LAYOUT = [
    ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '='],
    ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', '[', ']'],
    ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';', "'", '\\'],
    ['z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', '/', ' ', ' ']
]

KEY_TO_COORD = {}
for r, row in enumerate(LAYOUT):
    for c, key in enumerate(row):
        KEY_TO_COORD[key] = (r / 3.0, c / 11.0)

def get_coord(char):
    return KEY_TO_COORD.get(char, (0.0, 0.0))

# 2. Reconstruct Model Network Layout exactly as trained
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size + 2, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)
        
    def forward(self, x, coords):
        embedded = self.embedding(x)
        x_combined = torch.cat((embedded, coords), dim=2)
        out, (hidden, cell) = self.lstm(x_combined)
        hidden = torch.tanh(self.fc_hidden(torch.cat((hidden[0:2], hidden[2:4]), dim=2))) if hidden.shape[0] > 2 else hidden
        cell = torch.tanh(self.fc_cell(torch.cat((cell[0:2], cell[2:4]), dim=2))) if cell.shape[0] > 2 else cell
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, hidden, cell):
        x = x.unsqueeze(1)
        embedded = self.embedding(x)
        out, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        predictions = self.fc(out.squeeze(1))
        return predictions, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

# 3. Interactive Evaluator Definition
class KeyboardPredictor:
    def __init__(self, model_path, max_len=20):
        checkpoint = torch.load(model_path, map_location=device)
        self.char_to_idx = checkpoint['char_to_idx']
        self.idx_to_char = checkpoint['idx_to_char']
        self.max_len = max_len
        
        # Hyperparameters matching training script
        vocab_size = len(self.char_to_idx)
        encoder = Encoder(vocab_size, 64, 256, 2)
        decoder = Decoder(vocab_size, 64, 256, 2)
        
        self.model = Seq2Seq(encoder, decoder).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def predict_word_beam(self, word, beam_width=3):
        word = word.lower()
        if not word:
            return ""
            
        # 1. Prepare inputs exactly like before
        input_idxs = [self.char_to_idx.get(c, self.char_to_idx["<UNK>"]) for c in word][:self.max_len]
        input_coords = [get_coord(c) for c in word][:self.max_len]
        
        pad_len = self.max_len - len(input_idxs)
        input_idxs += [self.char_to_idx["<PAD>"]] * pad_len
        input_coords += [(0.0, 0.0)] * pad_len
        
        src_tensor = torch.tensor([input_idxs], dtype=torch.long).to(device)
        coords_tensor = torch.tensor([input_coords], dtype=torch.float32).to(device)
        
        with torch.no_grad():
            # Get initial encoder hidden states
            hidden, cell = self.model.encoder(src_tensor, coords_tensor)
            
            # Start token sequence
            start_token = self.char_to_idx["<SOS>"]
            
            # The Beam list will store tuples of: (log_probability, token_list, hidden_state, cell_state)
            # We initialize it with a log probability of 0.0
            beams = [(0.0, [start_token], hidden, cell)]
            
            for _ in range(self.max_len):
                all_candidates = []
                
                # Expand each current hypothesis in our beam
                for log_prob, tokens, h, c in beams:
                    # If a beam already hit the end token, preserve it as a candidate
                    if tokens[-1] == self.char_to_idx["<EOS>"]:
                        all_candidates.append((log_prob, tokens, h, c))
                        continue
                        
                    # Feed the last predicted token into the decoder
                    decoder_input = torch.tensor([tokens[-1]], dtype=torch.long).to(device)
                    output, next_h, next_c = self.model.decoder(decoder_input, h, c)
                    
                    # Convert raw model outputs (logits) to log probabilities
                    log_probs = torch.log_softmax(output, dim=1).squeeze(0)
                    
                    # Get the top K highest probability character tokens
                    topk_log_probs, topk_idxs = torch.topk(log_probs, beam_width)
                    
                    for i in range(beam_width):
                        next_token = topk_idxs[i].item()
                        next_log_prob = topk_log_probs[i].item()
                        
                        # Cumulative score calculation
                        all_candidates.append((
                            log_prob + next_log_prob, 
                            tokens + [next_token], 
                            next_h, 
                            next_c
                        ))
                
                # Sort all new paths by their total log probability score (highest first)
                ordered = sorted(all_candidates, key=lambda x: x[0], reverse=True)
                
                # Prune the list down to retain only the top K beams
                beams = ordered[:beam_width]
                
                # Safety check: if all active beams have reached the EOS token, we can break early
                if all(b[1][-1] == self.char_to_idx["<EOS>"] for b in beams):
                    break
            
            # Extract the winning sequence tokens from the top beam
            best_tokens = beams[0][1]
            
            # Reconstruct characters while ignoring structural control tokens
            predicted_chars = []
            for tok in best_tokens:
                if tok in [self.char_to_idx["<SOS>"], self.char_to_idx["<EOS>"], self.char_to_idx["<PAD>"]]:
                    continue
                predicted_chars.append(self.idx_to_char[tok])
                
        return "".join(predicted_chars)

    def predict_sentence(self, sentence):
        # Split line text entries by structural whitespace formatting tokens
        words = sentence.split(" ")
        decoded_words = [self.predict_word_beam(w) for w in words]
        return " ".join(decoded_words)

# 4. Interactive Execution Interface Environment Block
if __name__ == "__main__":
    print("Loading Local Keyboard Decryption Core Weights...")
    predictor = KeyboardPredictor("keyboard_lstm_model.pt")
    print("System active! Enter your shifted text below. Press Ctrl+C to exit.\n")
    
    while True:
        try:
            user_input = input("Shifted Gibberish > ")
            if user_input.strip() == "":
                continue
            prediction = predictor.predict_sentence(user_input)
            print(f"Decoded Intention  > {prediction}\n")
        except KeyboardInterrupt:
            print("\nExiting Engine Loop.")
=======
import torch
import torch.nn as nn
import numpy as np

# Set device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Re-declare the Keyboard Matrix for Spatial Feature Extraction
LAYOUT = [
    ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '='],
    ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', '[', ']'],
    ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';', "'", '\\'],
    ['z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', '/', ' ', ' ']
]

KEY_TO_COORD = {}
for r, row in enumerate(LAYOUT):
    for c, key in enumerate(row):
        KEY_TO_COORD[key] = (r / 3.0, c / 11.0)

def get_coord(char):
    return KEY_TO_COORD.get(char, (0.0, 0.0))

# 2. Reconstruct Model Network Layout exactly as trained
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size + 2, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)
        
    def forward(self, x, coords):
        embedded = self.embedding(x)
        x_combined = torch.cat((embedded, coords), dim=2)
        out, (hidden, cell) = self.lstm(x_combined)
        hidden = torch.tanh(self.fc_hidden(torch.cat((hidden[0:2], hidden[2:4]), dim=2))) if hidden.shape[0] > 2 else hidden
        cell = torch.tanh(self.fc_cell(torch.cat((cell[0:2], cell[2:4]), dim=2))) if cell.shape[0] > 2 else cell
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, hidden, cell):
        x = x.unsqueeze(1)
        embedded = self.embedding(x)
        out, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        predictions = self.fc(out.squeeze(1))
        return predictions, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

# 3. Interactive Evaluator Definition
class KeyboardPredictor:
    def __init__(self, model_path, max_len=20):
        checkpoint = torch.load(model_path, map_location=device)
        self.char_to_idx = checkpoint['char_to_idx']
        self.idx_to_char = checkpoint['idx_to_char']
        self.max_len = max_len
        
        # Hyperparameters matching training script
        vocab_size = len(self.char_to_idx)
        encoder = Encoder(vocab_size, 64, 256, 2)
        decoder = Decoder(vocab_size, 64, 256, 2)
        
        self.model = Seq2Seq(encoder, decoder).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def predict_word_beam(self, word, beam_width=3):
        word = word.lower()
        if not word:
            return ""
            
        # 1. Prepare inputs exactly like before
        input_idxs = [self.char_to_idx.get(c, self.char_to_idx["<UNK>"]) for c in word][:self.max_len]
        input_coords = [get_coord(c) for c in word][:self.max_len]
        
        pad_len = self.max_len - len(input_idxs)
        input_idxs += [self.char_to_idx["<PAD>"]] * pad_len
        input_coords += [(0.0, 0.0)] * pad_len
        
        src_tensor = torch.tensor([input_idxs], dtype=torch.long).to(device)
        coords_tensor = torch.tensor([input_coords], dtype=torch.float32).to(device)
        
        with torch.no_grad():
            # Get initial encoder hidden states
            hidden, cell = self.model.encoder(src_tensor, coords_tensor)
            
            # Start token sequence
            start_token = self.char_to_idx["<SOS>"]
            
            # The Beam list will store tuples of: (log_probability, token_list, hidden_state, cell_state)
            # We initialize it with a log probability of 0.0
            beams = [(0.0, [start_token], hidden, cell)]
            
            for _ in range(self.max_len):
                all_candidates = []
                
                # Expand each current hypothesis in our beam
                for log_prob, tokens, h, c in beams:
                    # If a beam already hit the end token, preserve it as a candidate
                    if tokens[-1] == self.char_to_idx["<EOS>"]:
                        all_candidates.append((log_prob, tokens, h, c))
                        continue
                        
                    # Feed the last predicted token into the decoder
                    decoder_input = torch.tensor([tokens[-1]], dtype=torch.long).to(device)
                    output, next_h, next_c = self.model.decoder(decoder_input, h, c)
                    
                    # Convert raw model outputs (logits) to log probabilities
                    log_probs = torch.log_softmax(output, dim=1).squeeze(0)
                    
                    # Get the top K highest probability character tokens
                    topk_log_probs, topk_idxs = torch.topk(log_probs, beam_width)
                    
                    for i in range(beam_width):
                        next_token = topk_idxs[i].item()
                        next_log_prob = topk_log_probs[i].item()
                        
                        # Cumulative score calculation
                        all_candidates.append((
                            log_prob + next_log_prob, 
                            tokens + [next_token], 
                            next_h, 
                            next_c
                        ))
                
                # Sort all new paths by their total log probability score (highest first)
                ordered = sorted(all_candidates, key=lambda x: x[0], reverse=True)
                
                # Prune the list down to retain only the top K beams
                beams = ordered[:beam_width]
                
                # Safety check: if all active beams have reached the EOS token, we can break early
                if all(b[1][-1] == self.char_to_idx["<EOS>"] for b in beams):
                    break
            
            # Extract the winning sequence tokens from the top beam
            best_tokens = beams[0][1]
            
            # Reconstruct characters while ignoring structural control tokens
            predicted_chars = []
            for tok in best_tokens:
                if tok in [self.char_to_idx["<SOS>"], self.char_to_idx["<EOS>"], self.char_to_idx["<PAD>"]]:
                    continue
                predicted_chars.append(self.idx_to_char[tok])
                
        return "".join(predicted_chars)

    def predict_sentence(self, sentence):
        # Split line text entries by structural whitespace formatting tokens
        words = sentence.split(" ")
        decoded_words = [self.predict_word_beam(w) for w in words]
        return " ".join(decoded_words)

# 4. Interactive Execution Interface Environment Block
if __name__ == "__main__":
    print("Loading Local Keyboard Decryption Core Weights...")
    predictor = KeyboardPredictor("keyboard_lstm_model.pt")
    print("System active! Enter your shifted text below. Press Ctrl+C to exit.\n")
    
    while True:
        try:
            user_input = input("Shifted Gibberish > ")
            if user_input.strip() == "":
                continue
            prediction = predictor.predict_sentence(user_input)
            print(f"Decoded Intention  > {prediction}\n")
        except KeyboardInterrupt:
            print("\nExiting Engine Loop.")
>>>>>>> 7435dcc28e45064eea611b3301881f712b6cc929
            break