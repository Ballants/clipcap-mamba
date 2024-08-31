import argparse

import matplotlib.pyplot as plt
from transformers import AdamW, get_cosine_schedule_with_warmup
from transformers.utils import logging

from model.model import Captioner
from utils.prepare_data import *

logging.set_verbosity_error()


def train(dataset_name, epochs, learning_rate):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Captioner(llm_model="state-spaces/mamba-130m-hf", d_clip=512, prefix_len=10, num_layers=8, device=device,
                embed_size=768).to(device)

    print('total number of parameters of the network is: ' + str(
        sum(p.numel() for p in model.parameters() if p.requires_grad)))

    dir_data = f'../data/{dataset_name}/'
    train_dl = torch.load(os.path.join(dir_data, f'train_dl.pt'))
    val_dl = torch.load(os.path.join(dir_data, f'val_dl.pt'))

    num_training_steps = epochs * len(train_dl)
    warmup_steps = int(num_training_steps * 9 / 100)  # 9% of the total training steps.

    checkpoint_path = f'../checkpoints/{dataset_name}/'
    create_directory_if_not_exists(checkpoint_path)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=learning_rate)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=num_training_steps)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    patience = 5
    patience_counter = 0

    print(f"Number of training steps: {num_training_steps}")
    print(f"Number of warmup steps: {warmup_steps}")

    # Training loop
    for epoch in range(1, epochs + 1):

        # TRAIN
        model.train()
        running_loss = 0.0
        len_train_dl = 0
        optimizer.zero_grad()  # ensure gradients are zeroed before starting accumulation

        for batch_idx, (clip_emb, target_cap, attention_mask) in enumerate(
                tqdm(train_dl, desc=f"Epoch: {epoch} | ", total=len(train_dl))):

            bs = clip_emb.shape[0]
            len_train_dl += bs

            clip_emb, target_cap, attention_mask = clip_emb.to(device), target_cap.to(device), attention_mask.to(device)

            loss = model(target_cap, clip_emb, attention_mask)

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            running_loss += loss.item()

            if batch_idx % 100 == 0:
                print('\tTrain loss: %.5f' % (running_loss / len_train_dl))
                for param_group in optimizer.param_groups:
                    print(f'lr : %.7f' % param_group['lr'])

        train_loss = running_loss / len_train_dl
        print('\tTraining loss: %.5f' % train_loss)

        # VAL
        model.eval()
        with torch.no_grad():
            running_loss = 0.0
            len_val_dl = 0
            for batch_idx, (clip_emb, target_cap, attention_mask) in enumerate(
                    tqdm(val_dl, desc=f"Epoch: {epoch} | Loss: ---", total=len(val_dl))):
                bs = clip_emb.shape[0]
                len_val_dl += bs

                clip_emb, target_cap, attention_mask = clip_emb.to(device), target_cap.to(device), attention_mask.to(
                    device)

                loss = model(target_cap, clip_emb, attention_mask)

                running_loss += loss.item()

            val_loss = running_loss / len_val_dl
            print('\tValidation loss: %.5f' % val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_path, f'best_ckp.pth'))
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Create a plot for training and validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epoch + 1), train_losses, label='Training Loss', marker='o')
    plt.plot(range(1, epoch + 1), val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # Save the loss plot to a file if needed
    if checkpoint_path is not None:
        loss_plot_path = os.path.join(checkpoint_path, f'loss_plot_epoch_{epoch}.png')
        plt.savefig(loss_plot_path)

    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate for training')

    args = parser.parse_args()

    train(args.dataset_name, args.epochs, args.lr)

if __name__ == '__main__':
    main()