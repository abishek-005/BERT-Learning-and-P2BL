import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import pandas as pd

# -----------------------------
# Create Dataset
# -----------------------------

data = {
    'text': [
        'I love this product!',
        'The service was terrible.',
        'This is amazing!',
        'I hate this company!',
        'Great experience!',
        'Worst product ever!',
        'Excellent customer service!',
        'The food was awful!',
        'I highly recommend this!',
        'Terrible experience!'
    ],
    'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

# -----------------------------
#  Train / Validation Split
# -----------------------------

train_text, val_text, train_labels, val_labels = train_test_split(
    df['text'],
    df['label'],
    random_state=42,
    test_size=0.2,
    stratify=df['label']
)


train_text = train_text.reset_index(drop=True)
val_text = val_text.reset_index(drop=True)
train_labels = train_labels.reset_index(drop=True)
val_labels = val_labels.reset_index(drop=True)

# -----------------------------
#  Custom Dataset Class
# -----------------------------

class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

# -----------------------------
#  Load Model & Tokenizer
# -----------------------------

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# -----------------------------
#  Create DataLoaders
# -----------------------------

batch_size = 8
max_len = 128

train_dataset = CustomDataset(train_text, train_labels, tokenizer, max_len)
val_dataset = CustomDataset(val_text, val_labels, tokenizer, max_len)

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=RandomSampler(train_dataset))
val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=SequentialSampler(val_dataset))

# -----------------------------
#  Setup Device & Optimizer
# -----------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# -----------------------------
# Training Loop
# -----------------------------

num_epochs = 3
best_accuracy = 0

for epoch in range(num_epochs):
    model.train()

    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)

        loss = outputs.loss
        loss.backward()
        optimizer.step()

    # -----------------------------
    # Validation
    # -----------------------------

    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)

            loss = outputs.loss
            total_loss += loss.item()

            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    epoch_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(true_labels, predictions)

    print(f"Epoch {epoch+1}: Loss = {epoch_loss:.4f}, Accuracy = {accuracy:.4f}")

    # Save Best Model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), 'best_model.pth')

# -----------------------------
#  Final Evaluation
# -----------------------------

model.load_state_dict(torch.load('best_model.pth'))
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask)

        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

print("\nFinal Classification Report:\n")
print(classification_report(y_true, y_pred))