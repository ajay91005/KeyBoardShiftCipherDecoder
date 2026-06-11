<<<<<<< HEAD
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training Matrix Core initialized on: {device}")

# ==========================================
# 1. CHARACTER VOCABULARY SETUP
# ==========================================
PAD_TOK, SOS_TOK, EOS_TOK, UNK_TOK = "<PAD>", "<SOS>", "<EOS>", "<UNK>"
LAYOUT_CHARS = list("abcdefghijklmnopqrstuvwxyz1234567890 -=[]\\;',./")
all_chars = [PAD_TOK, SOS_TOK, EOS_TOK, UNK_TOK] + LAYOUT_CHARS

char_to_idx = {char: idx for idx, char in enumerate(all_chars)}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

class SentenceAttentionDataset(Dataset):
    def __init__(self, csv_file, max_len=120):
        self.df = pd.read_csv(csv_file).dropna()
        self.max_len = max_len
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        gibberish = str(row['input_gibberish']).lower()
        correct = str(row['target_correct']).lower()
        
        # Build Input Sequence
        input_idxs = [char_to_idx.get(c, char_to_idx[UNK_TOK]) for c in gibberish][:self.max_len-1] + [char_to_idx[EOS_TOK]]
        # Build Target Sequence with boundary framing tokens
        target_idxs = [char_to_idx[SOS_TOK]] + [char_to_idx.get(c, char_to_idx[UNK_TOK]) for c in correct][:self.max_len-2] + [char_to_idx[EOS_TOK]]
        
        # Track raw lengths before padding for masking
        input_len = len(input_idxs)
        
        # Apply padding calculations
        input_idxs += [char_to_idx[PAD_TOK]] * (self.max_len - len(input_idxs))
        target_idxs += [char_to_idx[PAD_TOK]] * (self.max_len - len(target_idxs))
        
        return (
            torch.tensor(input_idxs, dtype=torch.long),
            torch.tensor(target_idxs, dtype=torch.long),
            torch.tensor(input_len, dtype=torch.long)
        )

# ==========================================
# 2. ATTENTION SELECTION ARCHITECTURE
# ==========================================
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 3, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        # hidden shape: (batch_size, hidden_size)
        # encoder_outputs shape: (batch_size, seq_len, hidden_size * 2)
        seq_len = encoder_outputs.shape[1]
        
        # Repeat decoder hidden state across all encoder timesteps
        hidden_expanded = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Compute energy scores matrix
        energy = torch.tanh(self.attn(torch.cat((hidden_expanded, encoder_outputs), dim=2)))
        attention_scores = self.v(energy).squeeze(2) # (batch_size, seq_len)
        
        return torch.softmax(attention_scores, dim=1)

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=char_to_idx[PAD_TOK])
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        
        # Projection layers to bridge Bidirectional (hidden_size * 2) -> Unidirectional (hidden_size)
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)
        
    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded)
        
        # hidden shape from bidirectional LSTM: (2, batch_size, hidden_size)
        # Concatenate forward and backward states along the feature dimension
        hidden_combined = torch.cat((hidden[0], hidden[1]), dim=1) # (batch_size, hidden_size * 2)
        cell_combined = torch.cat((cell[0], cell[1]), dim=1)   # (batch_size, hidden_size * 2)
        
        # Project down to match the decoder's hidden size configuration
        hidden = torch.tanh(self.fc_hidden(hidden_combined)) # (batch_size, hidden_size)
        cell = torch.tanh(self.fc_cell(cell_combined))       # (batch_size, hidden_size)
        
        return outputs, hidden, cell

class DecoderWithAttention(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=char_to_idx[PAD_TOK])
        self.attention = Attention(hidden_size)
        self.lstm = nn.LSTM((hidden_size * 2) + embed_size, hidden_size, batch_first=True)
        self.fc_out = nn.Linear(hidden_size * 3, vocab_size)
        
    def forward(self, x, hidden, cell, encoder_outputs):
        x = x.unsqueeze(1) # Make batch sequence tracking shape
        embedded = self.embedding(x)
        
        # Calculate dynamic attention weights vector
        attn_weights = self.attention(hidden, encoder_outputs).unsqueeze(1)
        context_vector = torch.bmm(attn_weights, encoder_outputs)
        
        # Combine hidden context matrix with the current character identity
        lstm_input = torch.cat((embedded, context_vector), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden.unsqueeze(0), cell.unsqueeze(0)))
        
        # Project raw scores down to our vocabulary sizing
        output_projection = torch.cat((output.squeeze(1), context_vector.squeeze(1)), dim=1)
        predictions = self.fc_out(output_projection)
        
        return predictions, hidden.squeeze(0), cell.squeeze(0)

class GlobalSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, source, target, teacher_force_ratio=0.4):
        batch_size = source.shape[0]
        target_len = target.shape[1]
        vocab_size = self.decoder.fc_out.out_features
        
        outputs = torch.zeros(batch_size, target_len, vocab_size).to(device)
        encoder_outputs, hidden, cell = self.encoder(source)
        
        decoder_input = target[:, 0] # Grab <SOS> token array
        
        for t in range(1, target_len):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell, encoder_outputs)
            outputs[:, t] = output
            
            best_guess = output.argmax(1)
            decoder_input = target[:, t] if np.random.random() < teacher_force_ratio else best_guess
            
        return outputs

# ==========================================
# 3. RUNTIME OPTIMIZATION PIPELINE
# ==========================================
def train():
    dataset = SentenceAttentionDataset("perfect_keyboard_sentences.csv")
    train_set, val_set = train_test_split(dataset, test_size=0.05, random_state=42)
    
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    
    # Target-scaled Hyperparameters
    encoder = Encoder(len(char_to_idx), 128, 512)
    decoder = DecoderWithAttention(len(char_to_idx), 128, 512)
    model = GlobalSeq2Seq(encoder, decoder).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=char_to_idx[PAD_TOK])
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    epochs = 8
    print("\nTraining execution started. Optimizing mapping vectors...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for src, trg, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            src, trg = src.to(device), trg.to(device)
            
            optimizer.zero_grad()
            output = model(src, trg)
            
            # Reshape tensors for cross-entropy calculations
            output = output[:, 1:].reshape(-1, output.shape[-1])
            trg = trg[:, 1:].reshape(-1)
            
            loss = criterion(output, trg)
            loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1} Completed. Sequence Loss: {epoch_loss/len(train_loader):.5f}")
        
    torch.save({
        'model_state': model.state_dict(),
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char
    }, "perfect_sentence_attention.pt")
    print("\nCheckpoint successfully verified and saved as 'perfect_sentence_attention.pt'!")

if __name__ == "__main__":
    train()
=======
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


LAYOUT = [
    ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '='],
    ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', '[', ']'],
    ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';', "'", '\\'],
    ['z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', '/', ' ', ' ']
]

# Create coordinate mapping
KEY_TO_COORD = {}
for r, row in enumerate(LAYOUT):
    for c, key in enumerate(row):
        KEY_TO_COORD[key] = (r / 3.0, c / 11.0)  # Normalize coordinates between 0 and 1

def get_coord(char):
    return KEY_TO_COORD.get(char, (0.0, 0.0))

# 2. Vocabulary Building
# Special tokens
PAD_TOK = "<PAD>"
SOS_TOK = "<SOS>"
EOS_TOK = "<EOS>"
UNK_TOK = "<UNK>"

all_chars = [PAD_TOK, SOS_TOK, EOS_TOK, UNK_TOK] + sorted(list(set([k for row in LAYOUT for k in row])))
char_to_idx = {char: idx for idx, char in enumerate(all_chars)}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

# 3. Custom Dataset Class
class KeyboardShiftDataset(Dataset):
    def __init__(self, csv_file, max_len=20):
        self.df = pd.read_csv(csv_file).dropna()
        self.max_len = max_len
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        gibberish = str(row['input_gibberish']).lower()
        correct = str(row['target_correct']).lower()
        
        # Build Input (Tokens + Spatial Coordinates)
        input_idxs = [char_to_idx.get(c, char_to_idx[UNK_TOK]) for c in gibberish][:self.max_len]
        input_coords = [get_coord(c) for c in gibberish][:self.max_len]
        
        # Build Target (With SOS and EOS tokens)
        target_idxs = [char_to_idx[SOS_TOK]] + [char_to_idx.get(c, char_to_idx[UNK_TOK]) for c in correct][:self.max_len-2] + [char_to_idx[EOS_TOK]]
        
        # Padding lengths
        input_pad_len = self.max_len - len(input_idxs)
        target_pad_len = self.max_len - len(target_idxs)
        
        # Apply padding
        input_idxs += [char_to_idx[PAD_TOK]] * input_pad_len
        input_coords += [(0.0, 0.0)] * input_pad_len
        target_idxs += [char_to_idx[PAD_TOK]] * target_pad_len
        
        return (
            torch.tensor(input_idxs, dtype=torch.long),
            torch.tensor(input_coords, dtype=torch.float32),
            torch.tensor(target_idxs, dtype=torch.long)
        )

# 4. Seq2Seq LSTM Model with Spatial Awareness
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=char_to_idx[PAD_TOK])
        # The input features combine character embeddings (embed_size) + 2D coordinates (2)
        self.lstm = nn.LSTM(embed_size + 2, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)
        
    def forward(self, x, coords):
        embedded = self.embedding(x)
        # Concatenate spatial coordinates along the feature dimension
        x_combined = torch.cat((embedded, coords), dim=2)
        out, (hidden, cell) = self.lstm(x_combined)
        
        # Collapse bidirectional states into single unidirectional states for the decoder
        hidden = torch.tanh(self.fc_hidden(torch.cat((hidden[0:2], hidden[2:4]), dim=2))) if hidden.shape[0] > 2 else hidden
        cell = torch.tanh(self.fc_cell(torch.cat((cell[0:2], cell[2:4]), dim=2))) if cell.shape[0] > 2 else cell
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=char_to_idx[PAD_TOK])
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, hidden, cell):
        # Shape of x: (Batch_Size), we make it (Batch_Size, 1) for sequence processing
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
        
    def forward(self, source, coords, target, teacher_force_ratio=0.5):
        batch_size = source.shape[0]
        target_len = target.shape[1]
        vocab_size = self.decoder.fc.out_features
        
        outputs = torch.zeros(batch_size, target_len, vocab_size).to(device)
        hidden, cell = self.encoder(source, coords)
        
        # Grab initial SOS token
        x = target[:, 0]
        
        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[:, t] = output
            best_guess = output.argmax(1)
            x = target[:, t] if np.random.random() < teacher_force_ratio else best_guess
            
        return outputs

# 5. Execution Logic
def train_model():
    print("Loading data...")
    dataset = KeyboardShiftDataset("keyboard_shift_dataset.csv")
    train_set, val_set = train_test_split(dataset, test_size=0.1, random_state=42)
    
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
    
    # Model Hyperparameters
    VOCAB_SIZE = len(char_to_idx)
    EMBED_SIZE = 64
    HIDDEN_SIZE = 256
    NUM_LAYERS = 2
    
    encoder = Encoder(VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS)
    decoder = Decoder(VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS)
    model = Seq2Seq(encoder, decoder).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=char_to_idx[PAD_TOK])
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    epochs = 5
    print("Beginning Training Routine...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (src, coords, trg) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            src, coords, trg = src.to(device), coords.to(device), trg.to(device)
            
            optimizer.zero_grad()
            output = model(src, coords, trg)
            
            # Reshape output for loss calculation (flatten batch and sequence lengths)
            output = output[:, 1:].reshape(-1, output.shape[-1])
            trg = trg[:, 1:].reshape(-1)
            
            loss = criterion(output, trg)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1} Complete. Average Loss: {epoch_loss/len(train_loader):.4f}")
        
    # Save the states
    torch.save({
        'model_state_dict': model.state_dict(),
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char
    }, "keyboard_lstm_model.pt")
    print("Model saved successfully as 'keyboard_lstm_model.pt'!")

if __name__ == "__main__":
    train_model()
>>>>>>> 7435dcc28e45064eea611b3301881f712b6cc929
