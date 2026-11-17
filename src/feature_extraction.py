import torch
import numpy as np
from tqdm.auto import tqdm

from torch.utils.data import DataLoader

def extract_embeddings(dataset, model, batch_size=16):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"]
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size
    )

    all_embeddings = []
    all_labels = []

    with torch.inference_mode():

        for batch in tqdm(dataloader, desc="Extracting embeddings"):

            input_ids = batch["input_ids"].to(device)

            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # Extract CLS representation
            embeddings = outputs.last_hidden_state[:, 0, :]

            all_embeddings.append(
                embeddings.cpu().numpy()
            )

            all_labels.append(
                batch["label"].numpy()
            )

    X = np.concatenate(all_embeddings, axis=0)

    y = np.concatenate(all_labels, axis=0)

    return X, y