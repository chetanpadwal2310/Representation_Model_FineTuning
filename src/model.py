from transformers import AutoModel

model_name = "distilbert/distilbert-base-uncased"

def load_pretrained_model():
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return model
    