# deep learning libraries
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

# other libraries
from tqdm.auto import tqdm

# own modules
from src.layers import CNNModel
from src.data import load_data
from src.utils import (
    Accuracy,
    save_model,
    set_seed,
)

# set device
device: torch.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

# set all seeds and set number of threads
set_seed(42)
torch.set_num_threads(8)

# static variables
DATA_PATH: str = "data"

NUMBER_OF_CLASSES: int = 10



def main() -> None:
    """
    This function is the main program for the training.
    """

    # TODO

    #hyperparameters
    epochs: int = 50
    lr: float = 5e-4
    batch_size: int = 32
    hidden_sizes: tuple[int, ...] = (32, 64, 128)
    #empty nohup.out file

    open("nohup.out", "w").close()

    #load data

    train_loader:DataLoader
    valid_loader:DataLoader
    train_loader, valid_loader, test_loader = load_data(DATA_PATH, batch_size=batch_size)
    #define name and writer
    name: str = f"model_lr_{lr}_hs_{hidden_sizes}_{batch_size}_{epochs}"
    writer: SummaryWriter = SummaryWriter(f"runs/{name}")  

    #define model

    inputs: torch.Tensor = next(iter(train_loader))[0]
    outputs: torch.Tensor = next(iter(train_loader))[1]
    print(device)

    NUMBER_OF_CLASSES: int = outputs.shape[0]
    
    model: CNNModel = CNNModel(
        input_shape=inputs.shape[1:],
        output_shape=NUMBER_OF_CLASSES
    ).to(device)

    #define loss and optimizer
    loss: torch.nn.Module = torch.nn.CrossEntropyLoss()
    optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=lr) 

    #train loop
    for epoch in tqdm(range(epochs)):
        
        #define metric list
        losses: list[float] = []
        accuracies: list[float] = []

        for inputs, labels in train_loader:

            #pass to device
            inputs, labels = inputs.to(device), labels.to(device)
            
            #compute outputs and loss
            outputs: torch.Tensor = model(inputs) #y
            loss_value = loss(outputs, labels.long())

            #zero gradients
            optimizer.zero_grad() #reset gradients so that they do not accumulate
            loss_value.backward()
            optimizer.step()

            #add metrics to the list
            losses.append(loss_value.item())

            #compute accuracy
            accuracy = Accuracy()
            accuracy.update(outputs, labels)
            accuracies.append(accuracy.compute())   
        
        #write to tensorboard
        writer.add_scalar("Loss/train", np.mean(losses), epoch)
        writer.add_scalar("Accuracy/train", np.mean(accuracies), epoch)

        #validation loop
        with torch.no_grad():
            losses = []

            for inputs, labels in valid_loader:

                #pass to device
                inputs, labels = inputs.to(device), labels.to(device)
                #compute outputs and loss
                outputs = model(inputs)
                loss_value = loss(outputs, labels.long())

                #add metrics to the list
                losses.append(loss_value.item())

                #compute accuracy
                accuracy = Accuracy()
                accuracy.update(outputs, labels)
                accuracies.append(accuracy.compute())

            #write to tensorboard
            writer.add_scalar("Loss/valid", np.mean(losses), epoch)
            writer.add_scalar("Accuracy/valid", np.mean(accuracies), epoch)
        
        print(f"Epoch: {epoch}, Loss: {np.mean(losses)}, Accuracy: {np.mean(accuracies)}")

    #save model
    save_model(model, name)

    return None


if __name__ == "__main__":
    main()
