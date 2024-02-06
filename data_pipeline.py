import os
import re
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
from nltk.tokenize import word_tokenize


class ImageData(Dataset):
    def __init__(self, image_path, image_size=224):
        self.image_size = image_size
        self.image_path = [os.path.join(image_path, x) for x in os.listdir(image_path)]

        self.transforms = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        image_path = self.image_path[index]
        image = Image.open(image_path).convert("RGB")
        image = self.transforms(image)
        return image


class TextData(Dataset):
    def __init__(self, text_path):
        df = pd.read_csv(text_path)
        df.drop_duplicates(subset="image", inplace=True)
        df.reset_index(drop=True, inplace=True)
        self.sentences = df["caption"].apply(lambda x: re.sub(r"[^a-zA-Z0-9\s]", "", x).lower())

        self.word2index = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.index2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}

        for sentence in self.sentences:
            for index, word in enumerate(word_tokenize(sentence), start=4):
                self.word2index[word] = index
                self.index2word[index] = word

    def sentence_to_sequence(self, sentence):
        sequence = [
            self.word2index[token] if token in self.word2index
            else self.word2index["<UNK>"]
            for token in word_tokenize(sentence)]

        sequence = [self.word2index["<SOS>"]] + sequence + [self.word2index["<EOS>"]]
        return torch.tensor(sequence)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        sentence = self.sentences[index]
        indexed = self.sentence_to_sequence(sentence)
        return indexed


class CombinedDataset(Dataset):
    def __init__(self, image_data_path, text_data_path):
        self.image_dataset = ImageData(image_data_path)
        self.text_dataset = TextData(text_data_path)
        self.max_length = self.calculate_max_length()

    def vocab_size(self):
        return len(self.text_dataset.word2index)

    def calculate_max_length(self):
        max_len = 0
        for i in range(len(self.text_dataset)):
            text_item = self.text_dataset[i]
            max_len = max(max_len, len(text_item))
        return max_len

    @staticmethod
    def collate_fn(batch):
        img, txt = zip(*batch)
        txt = pad_sequence(txt, padding_value=0)
        return torch.stack(img), txt

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, index):
        image_item = self.image_dataset[index]
        text_item = self.text_dataset[index]
        return image_item, text_item


if __name__ == "__main__":
    dataset = CombinedDataset(
        image_data_path="Sample Data/Images",
        text_data_path="Sample Data/captions.txt"
        )
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=4,
        pin_memory=True
    )
    image, caption = next(iter(dataloader))
    print("Vocab Size: ", dataset.vocab_size())
    print("Image Shape: ", image.shape, "Dim:", image.ndim)
    print("Caption Shape: ", caption.shape, "Dim:", caption.ndim)
    print(dataset.max_length)