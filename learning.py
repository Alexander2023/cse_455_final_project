import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch import nn, optim, no_grad, device, cuda, save

from custom_transforms import convert_grayscale_to_rgb

DATA_PATH = "." # update to path of directory containing fer2013 folder, as needed
SAVED_WEIGHTS_PATH = "./saved_weights.pt"
BATCH_SIZE = 32
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 0.001
EPOCHS = 3
CLASSES = 7
PRETRAINED_WEIGHTS = models.ResNet18_Weights.DEFAULT
MODEL_FORMAT_TRANSFORMS_LIST = [transforms.Lambda(convert_grayscale_to_rgb),
                                PRETRAINED_WEIGHTS.transforms()]

def compute_correct_predictions(predictions, input_labels):
  # used from reference
  # https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
  _, predicted = torch.max(predictions.data, 1)
  return (predicted == input_labels).sum().item()

def prepare_loaders():
  transform = transforms.Compose(MODEL_FORMAT_TRANSFORMS_LIST)

  training_data = datasets.FER2013(DATA_PATH, "train", transform)
  testing_data = datasets.FER2013(DATA_PATH, "test", transform)

  return DataLoader(training_data, BATCH_SIZE, True), DataLoader(testing_data,
      BATCH_SIZE, True)

def train_model(available_device: device, model, training_loader: DataLoader, loss_function):
  model.train()
  optimizer = optim.SGD(model.parameters(), LEARNING_RATE, MOMENTUM,
      weight_decay=WEIGHT_DECAY)
  num_total = 0
  num_correct = 0
  for i in range(EPOCHS):
    print(f"Epoch #{i}")
    for _, (input_data, input_labels) in enumerate(training_loader):
      input_data = input_data.to(available_device)
      input_labels = input_labels.to(available_device)

      optimizer.zero_grad()
      predictions = model(input_data)
      loss = loss_function(predictions, input_labels)
      loss.backward()
      optimizer.step()

      print("Loss = ", loss.item())

      num_total += BATCH_SIZE
      num_correct += compute_correct_predictions(predictions, input_labels)

    print("Training Accuracy = ", num_correct / num_total)

def testing_model(available_device: device, model, testing_loader: DataLoader, loss_function):
  model.eval()
  num_total = 0
  num_correct = 0
  for input_data, input_labels in testing_loader:
    input_data = input_data.to(available_device)
    input_labels = input_labels.to(available_device)

    with no_grad():
      predictions = model(input_data)
      loss = loss_function(predictions, input_labels)

      print("Loss = ", loss.item())

      num_total += BATCH_SIZE
      num_correct += compute_correct_predictions(predictions, input_labels)

  print("Testing Accuracy = ", num_correct / num_total)

def main():
  available_device = device("cuda") if cuda.is_available() else device("cpu")
  model = models.resnet18(weights=PRETRAINED_WEIGHTS)
  model.fc = nn.Linear(model.fc.in_features, CLASSES)
  model.to(available_device)
  loss_function = nn.CrossEntropyLoss()

  print("Begin loading data")
  training_loader, testing_loader = prepare_loaders()
  print("Loading data completed")

  print("Begin training model")
  train_model(available_device, model, training_loader, loss_function)
  save(model.state_dict(), SAVED_WEIGHTS_PATH)
  print("Training model completed")

  print("Begin testing model")
  testing_model(available_device, model, testing_loader, loss_function)
  print("Testing model completed")

if __name__ == "__main__":
  main()