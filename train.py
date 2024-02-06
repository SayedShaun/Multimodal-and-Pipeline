import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from vision_model import VisionModel
from data_pipeline import CombinedDataset
from torch.nn.utils import clip_grad_norm_

dataset = CombinedDataset(
    image_data_path="Sample Data/Images",
    text_data_path="Sample Data/captions.txt"
    )
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=dataset.collate_fn
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embed_size = 300
hidden_size = 128
vocab_size = dataset.vocab_size()
num_layers = 2
model = VisionModel(embed_size, hidden_size, vocab_size, num_layers).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 20
for epoch in tqdm(range(epochs)):
    model.train()
    total_loss = 0.0
    for batch, (images, captions) in enumerate(dataloader):
        images = images.to(device)
        captions = captions.to(device)

        optimizer.zero_grad()
        outputs = model(images, captions[:-1])
        loss = loss_fn(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))

        loss.backward()
        optimizer.step()
        clip_grad_norm_(model.parameters(), max_norm=1)

        total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch + 1}/{epochs}], Average Loss: {average_loss}")
print("Training Finished")
