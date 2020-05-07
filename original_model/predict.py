import argparse
import os
import pandas as pd
import torch
import torchvision
import models


PATH_MODEL_480 = 'resnext50_480.pth'
BATCH_SIZE_480 = 1


class TestAntispoofDataset(torch.utils.data.dataset.Dataset):
    def __init__(
            self, paths, transform=None,
            loader=torchvision.datasets.folder.default_loader):
        self.paths = paths
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        image_info = self.paths[index]

        img = self.loader(image_info['path'])
        if self.transform is not None:
            img = self.transform(img)

        return image_info['id'], image_info['frame'], img

    def __len__(self):
        return len(self.paths)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path-images-csv', type=str, default='sample_data.csv')
    parser.add_argument('--path-test-dir', type=str, default='')
    parser.add_argument('--path-submission-csv', type=str, default='sample_out.csv')
    args = parser.parse_args()

    # prepare image paths
    test_dataset_paths = pd.read_csv(args.path_images_csv)
    path_test_dir = args.path_test_dir

    paths = [
        {
            'id': row.id,
            'frame': row.frame,
            'path': os.path.join(path_test_dir, row.path)
        } for _, row in test_dataset_paths.iterrows()]

    # prepare dataset and loader
    data_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(480),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    image_dataset = TestAntispoofDataset(
        paths=paths, transform=data_transforms)
    dataloader = torch.utils.data.DataLoader(
        image_dataset, batch_size=1, shuffle=False, num_workers=4)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load model
    model_480 = models.resnext50_32x4d()
    model_480.fc = torch.nn.Linear(model_480.fc.in_features, 1)
    checkpoint = torch.load(PATH_MODEL_480)
    model_480.load_state_dict(checkpoint['state_dict'])
    #    model.load_state_dict(torch.load(PATH_MODEL, map_location=device))
    model_480 = model_480.to(device)
    model_480.eval()

    # predict
    samples, frames, probabilities = [], [], []

    with torch.no_grad():
        for video, frame, batch in dataloader:
            batch = batch.to(device)
            from_model = model_480(batch).view(-1)
            print(from_model.cpu().numpy())
            probability = torch.sigmoid(from_model)

            samples.extend(video.numpy())
            frames.extend(frame.numpy())
            probabilities.extend(probability.cpu().numpy())

    # save
    predictions = pd.DataFrame.from_dict({
        'id': samples,
        'frame': frames,
        'probability': probabilities})
    print(predictions)
    predictions = predictions.groupby('id').probability.mean().reset_index()
    predictions['prediction'] = predictions.probability
   
    predictions[['id', 'prediction']].to_csv(
        args.path_submission_csv, index=False)
    
    
