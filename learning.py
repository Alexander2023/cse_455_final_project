import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms, models
from torch import nn, optim, no_grad, device, cuda, save, Generator

from custom_transforms import convert_grayscale_to_rgb, random_horizontal_flip, random_rotation

DATA_PATH = "." # update to path of directory containing fer2013 folder, as needed
SAVED_WEIGHTS_PATH = "./saved_weights.pt"
BATCH_SIZE = 32
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 0.001
EPOCHS = 3
CLASSES = 7
PRETRAINED_WEIGHTS = models.ResNet18_Weights.DEFAULT
MODEL_FORMAT_TRANSFORMS = [transforms.Lambda(convert_grayscale_to_rgb),
                           PRETRAINED_WEIGHTS.transforms()]
MODEL_FORMAT_TRANSFORMS_WITH_AUGMENTATION = [transforms.Lambda(random_horizontal_flip),
                                             transforms.Lambda(random_rotation),
                                             *MODEL_FORMAT_TRANSFORMS]
K_FOLDS = 2
USE_K_FOLDS = True # when False, holdout is used

def compute_correct_predictions(predictions, input_labels):
  # used from reference
  # https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
  _, predicted = torch.max(predictions.data, 1)
  return (predicted == input_labels).sum().item()

def prepare_loaders():
  transform = transforms.Compose(MODEL_FORMAT_TRANSFORMS_WITH_AUGMENTATION)
  training_data = datasets.FER2013(DATA_PATH, "train", transform)
  # using training for both train/test since Kaggle test set is unlabeled
  data_split = random_split(training_data, [0.8, 0.2])
  return DataLoader(data_split[0], BATCH_SIZE, True), DataLoader(data_split[1],
      BATCH_SIZE, True)

def get_k_data():
  transform = transforms.Compose(MODEL_FORMAT_TRANSFORMS_WITH_AUGMENTATION)
  training_data = datasets.FER2013(DATA_PATH, "train", transform)
  # ensures same splits for each k-fold cross-validation instance
  generator = Generator().manual_seed(0)
  k_split = random_split(training_data, [1.0 / K_FOLDS] * K_FOLDS, generator)
  return k_split, training_data

def prepare_k_loaders(k_data, testing_idx):
  k_split, training_data = k_data
  k_minus_1_training_indices = []
  for i, dataset in enumerate(k_split):
    if i != testing_idx:
      k_minus_1_training_indices.extend(dataset.indices)
  return (DataLoader(Subset(training_data, k_minus_1_training_indices), BATCH_SIZE, True),
          DataLoader(k_split[testing_idx], BATCH_SIZE, True))

def train_model(available_device: device, model, training_loader: DataLoader, loss_function):
  print("Begin training model")
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

      # training steps used from reference, with slight modification
      # https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
      optimizer.zero_grad()
      predictions = model(input_data)
      loss = loss_function(predictions, input_labels)
      loss.backward()
      optimizer.step()

      print("Loss = ", loss.item())

      num_total += BATCH_SIZE
      num_correct += compute_correct_predictions(predictions, input_labels)

    print("Training Accuracy = ", num_correct / num_total)
  print("Training model completed")

def testing_model(available_device: device, model, testing_loader: DataLoader, loss_function):
  print("Begin testing model")
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

  accuracy = num_correct / num_total
  print("Testing Accuracy = ", accuracy)
  print("Testing model completed")
  return accuracy

def main():
  available_device = device("cuda") if cuda.is_available() else device("cpu")
  loss_function = nn.CrossEntropyLoss()

  def prepare_model():
    model = models.resnet18(weights=PRETRAINED_WEIGHTS)
    model.fc = nn.Linear(model.fc.in_features, CLASSES)
    model.to(available_device)
    return model

  if USE_K_FOLDS:
    k_data = get_k_data()
    accuracy_sum = 0.0
    for k in range(K_FOLDS):
      print(f"k fold #{k}")
      model = prepare_model()
      training_loader, testing_loader = prepare_k_loaders(k_data, k)
      train_model(available_device, model, training_loader, loss_function)
      accuracy = testing_model(available_device, model, testing_loader, loss_function)
      accuracy_sum += accuracy
    print("Average validation accuracy = ", accuracy_sum / K_FOLDS)
  else:
    model = prepare_model()
    training_loader, testing_loader = prepare_loaders()
    train_model(available_device, model, training_loader, loss_function)
    save(model.state_dict(), SAVED_WEIGHTS_PATH)
    testing_model(available_device, model, testing_loader, loss_function)

if __name__ == "__main__":
  main()