import torch
from torch.utils.data import DataLoader, Dataset
import os

class customData(Dataset):
    def __init__(self, data):
        """
        [(X0, y0), (X1, y1), ....]
        """
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Here lets say we have x and y, 
        we load the x from the path, convert y to nums, and return
        think of it as a playlist now
        """
        path, y = self.data[idx]

        #song_featues = load from the path but we take randn values for now

        song_features = torch.randn(128)
        label = 1 if y == "rock" else 0

        return song_features, label
    
Data = [("rock_song1", "rock"), ("pop_song1", "pop"),
    ("rock_song2", "rock"), ("pop_song2", "pop"),
    ("rock_song3", "rock"), ("pop_song3", "pop")
]

dataset = customData(Data)

print(f"Len of dataset {len(dataset)}")
print(f"First element {dataset[0]}")



dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0
)

print("Here the data gets loaded")

for batch_idx, (songs, genres) in enumerate(dataloader):
    print(f"Batch {batch_idx}")
    print(f"Song shape {songs.shape}")
    print(f"Genres{genres}")

    if batch_idx >= 2:
        break # show only 2


# Now I am using gpt to learn for practice, here is the homework

posts_data = [
    ("Just had the best coffee ever! ☕", "happy"),
    ("Ugh, Monday morning blues 😴", "sad"),
    ("EXCITED for the weekend! 🎉", "happy"),
    ("Can't believe this happened to me 😠", "angry"),
    ("Feeling grateful for friends 💕", "happy"),
    ("This traffic is killing me 🚗", "angry"),
    ("Rainy days make me thoughtful 🌧️", "sad"),
    ("BEST DAY EVER! 🚀", "happy"),
    ("Why is everything going wrong? 😢", "sad")
]


class HomeworkDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return (len(self.data))
    
    def __getitem__(self, index):
        text, sentiment = self.data[index]

        # Now we tokenize the text in realcase, here we just do torch.randn
        X = torch.randn(128)
        
        #Ordinal encoding, can use ordinal encoder for it but
        #too tired right now, its 12:20 in the night chill
        if sentiment == "happy":
            y = 0
        elif sentiment == "sad":
            y = 1
        else:
            y = 2
        
        return X, y


print("Homework here")
homeworkdataset = HomeworkDataset(posts_data)

HomeworkDataloader = DataLoader(
    homeworkdataset,
    batch_size=4,
    num_workers=0,
    drop_last=True,
    shuffle=True
)

for batch_idx, (X, y) in enumerate(HomeworkDataloader):
    print(f"Batch {batch_idx}")
    print(f"Embedding shape: {X.shape}")
    print(f"labels: {y}")










#NOTES



DataLoader(
    dataset,
    batch_size=32,      # How many items per batch (like playlist size)
    shuffle=True,       # Mix up order each epoch (prevent overfitting)
    num_workers=4,      # Parallel loading (faster but uses more CPU)
    drop_last=True,     # Drop incomplete last batch (keeps batch size consistent)
    pin_memory=True     # Faster GPU transfer (if using CUDA)
)