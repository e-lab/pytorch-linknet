import cv2
import torch
import numpy as np

# Define color scheme
color_map = np.array([
    [0, 0, 0],        # Unlabled
    [128, 64, 128],   # Road
    [244, 35, 232],   # Sidewalk
    [70, 70, 70],     # Building
    [102, 102, 156],  # Wall
    [190, 153, 153],  # Fence
    [153, 153, 153],  # Pole
    [250, 170, 30],   # Traffic light
    [220, 220, 0],    # Traffic signal
    [107, 142, 35],   # Vegetation
    [152, 251, 152],  # Terrain
    [70, 130, 180],   # Sky
    [220, 20, 60],    # Person
    [255, 0, 0],      # Rider
    [0, 0, 142],      # Car
    [0, 0, 70],       # Truck
    [0, 60, 100],     # Bus
    [0, 80, 100],     # Train
    [0, 0, 230],      # Motorcycle
    [119, 11, 32]     # Bicycle
], dtype=np.uint8)

# Load model
m = torch.load('/media/HDD2/Models/abhi/1/model_resume.pth')
model = torch.nn.DataParallel(m['model_def'](20))
model.load_state_dict(m['state_dict'])
model.cuda()

root_dir = '/media/HDD1/Datasets/cityscapes/leftImg8bit_demoVideo/leftImg8bit/demoVideo/stuttgart_00/'
idx = 1
while idx <= 599:
    # Load image, resize and convert into a 'batchified' cuda tensor
    filename = '{}stuttgart_00_000000_{:06d}_leftImg8bit.png'.format(root_dir, idx)
    x = cv2.imread(filename)
    x = cv2.resize(x, None, fx=0.5, fy=0.5)
    input_image = torch.from_numpy(cv2.cvtColor(x, cv2.COLOR_BGR2RGB).transpose(2, 0, 1))/255
    input_image = input_image.unsqueeze(0).float().cuda()

    # Get neural network output
    y = model(torch.autograd.Variable(input_image))
    y = y.squeeze()
    pred = y.data.cpu().numpy()

    # Calculate prediction and colorized segemented output
    prediction = np.argmax(pred, axis=0)
    num_classes = 20
    pred_map = np.array(x) * 0
    for i in range(num_classes):
        mask = (prediction == i).astype(np.uint8)             # 1s at detection
        mask3D = np.repeat(mask[:, :, np.newaxis], 3, axis=2) # HxW -> HxWx3
        pred_map += mask3D * color_map[i]

    x_RGB = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    pred_map_RGB = cv2.cvtColor(pred_map, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(x_RGB, 0.5, pred_map_RGB, 0.5, 0)
    cv2.imshow('Original Image', x_RGB)
    cv2.imshow('Segmented Output', pred_map_RGB)
    cv2.imshow('Overlayed Image', overlay)

    idx += 1
    if cv2.waitKey(1) == 27: # ESC to stop
        break

