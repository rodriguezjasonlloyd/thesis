from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

from modules.data import get_data_loaders
from modules.model import build_model
from modules.trainer import seed_all, train_model

if __name__ == "__main__":
    seed_all(42)

    train_data_loader, val_data_loader, dataset = get_data_loaders(
        max_items_per_class=5, batch_size=4
    )

    model = build_model()
    optimizer = AdamW(model.parameters())
    criterion = CrossEntropyLoss()

    train_model(model, optimizer, criterion, train_data_loader, val_data_loader, 5)
