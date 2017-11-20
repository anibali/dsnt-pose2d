# Basic DSNT usage guide

```python results='hidden'
import torch
from torch import nn, optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import scipy.misc
import dsnt.nn

torch.manual_seed(12345)
```

## Building a coordinate regression model

The bulk of the model can be any sort of fully convolutional network (FCN).
Here we'll just use a custom network with three convolutional layers.

```python
class FCN(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.layers(x)
```

Using the DSNT layer, we can very simply extend any FCN to tackle
coordinate regression tasks.

```python
class CoordRegressionNetwork(nn.Module):
    def __init__(self, n_locations):
        super().__init__()
        self.fcn = FCN()
        self.hm_conv = nn.Conv2d(16, n_locations, kernel_size=1, bias=False)

    def forward(self, images):
        # 1. Run the images through our FCN
        fcn_out = self.fcn(images)
        # 2. Use a 1x1 conv to get one unnormalized heatmap per location
        unnormalized_heatmaps = self.hm_conv(fcn_out)
        # 3. Normalize the heatmaps
        heatmaps = dsnt.nn.softmax_2d(unnormalized_heatmaps)
        # 4. Calculate the coordinates
        coords = dsnt.nn.dsnt(heatmaps)

        return coords, heatmaps
```

## Training the model

To demonstrate the model in action, we're going to train on an image of a
raccoon's eye.

```python width='300px'
raccoon_face = scipy.misc.imresize(scipy.misc.face()[200:400, 600:800, :], (40, 40))
eye_x, eye_y = 24, 26

plt.imshow(raccoon_face)
plt.scatter([eye_x], [eye_y], color='red', marker='X')
plt.show()
```

The input and target need to be put into PyTorch tensors. Importantly,
the target coordinates are normalized so that they are in the range (-1, 1).
The DSNT layer always outputs coordinates in this range.

```python
raccoon_face_tensor = torch.from_numpy(raccoon_face).float().div(255).permute(2, 0, 1).unsqueeze(0)
eye_coords = torch.FloatTensor([[eye_x, eye_y]]).div(40 / 2).sub(1)

print('Target: {:0.4f}, {:0.4f}'.format(*list(eye_coords[0])))

input_var = Variable(raccoon_face_tensor, requires_grad=False).cuda()
target_var = Variable(eye_coords, requires_grad=False).cuda()
```

The coordinate regression model needs to be told ahead of time how many
locations to predict coordinates for. In this case we'll intantiate a
model to output 1 location per image.

```python
model = CoordRegressionNetwork(n_locations=1).cuda()
```

Doing a forward pass with the model is the same as with any PyTorch model.
The results aren't very good yet since the model is completely untrained.

```python width='300px'
coords, heatmaps = model(input_var)

print('Initial prediction: {:0.4f}, {:0.4f}'.format(*list(coords.data[0, 0])))
plt.imshow(heatmaps[0, 0].data.cpu().numpy())
plt.show()
```

Now we'll train the model to overfit the location of the eye. Of course,
for real applications the model should be trained and evaluated using
separate training and validation datasets!

```python width='300px'
optimizer = optim.RMSprop(model.parameters(), lr=2.5e-4)

for i in range(400):
    # Forward pass
    coords, heatmaps = model(input_var)

    # This value corresponds to 1 pixel standard deviation
    sigma = 2.0 / heatmaps.size(-1)
    # Euclidean loss with regularization
    loss = dsnt.nn.euclidean_loss(coords, target_var) + dsnt.nn.js_reg_loss(heatmaps, target_var, sigma)

    # Calculate gradients
    optimizer.zero_grad()
    loss.backward()

    # Update model parameters with RMSprop
    optimizer.step()

# Predictions after training
print('Predicted coords: {:0.4f}, {:0.4f}'.format(*list(coords.data[0, 0])))
plt.imshow(heatmaps[0, 0].data.cpu().numpy())
plt.show()
```
