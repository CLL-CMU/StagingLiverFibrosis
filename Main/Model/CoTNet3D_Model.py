import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the 3D CoTNet Layer
class CoTNetLayer3D(nn.Module):
    def __init__(self, dim=512, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        
        # Key embedding layer
        self.key_embed = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=kernel_size, padding=1, stride=1, bias=False),
            nn.BatchNorm3d(dim),
            nn.ReLU()
        )
        
        # Value embedding layer
        self.value_embed = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(dim)
        )
        
        # Attention embedding layer
        factor = 4
        self.attention_embed = nn.Sequential(
            nn.Conv3d(2 * dim, 2 * dim // factor, 1, bias=False),
            nn.BatchNorm3d(2 * dim // factor),
            nn.ReLU(),
            nn.Conv3d(2 * dim // factor, kernel_size * kernel_size * dim, 1, stride=1)
        )

    def forward(self, x):
        bs, c, d, h, w = x.shape
        k1 = self.key_embed(x)
        flatten1 = nn.Flatten(start_dim=2, end_dim=-1)
        v = flatten1(self.value_embed(x))
        y = torch.cat([k1, x], dim=1)
        att = self.attention_embed(y)
        att = att.view(bs, c, self.kernel_size * self.kernel_size, d, h, w)
        att = att.mean(2, keepdim=False)
        att = flatten1(att)
        
        k2 = F.softmax(att, dim=-1) * v
        k2 = k2.view(bs, c, d, h, w)
        return k1 + k2

# Define the 3D Bottleneck Layer
class Bottleneck3D(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck3D, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.cot_layer = CoTNetLayer3D(dim=planes, kernel_size=3)  # CoTNetLayer adapted for 3D
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
        # Average pooling if stride > 1
        if stride > 1:
            self.avd = nn.AvgPool3d(kernel_size=3, stride=2, padding=1)
        else:
            self.avd = None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Apply average pooling if avd is not None
        if self.avd is not None:
            out = self.avd(out)
        
        out = self.cot_layer(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        return out

# Define the CoTNet3D Network
class CoTNet3D(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(CoTNet3D, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1 = nn.Linear(512 * block.expansion, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

def CoTNet3D_Model(**kwargs):
    model = CoTNet3D(Bottleneck3D, [3, 4, 6, 3], **kwargs)
    return model

# Function to get the total number of parameters in the model
def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp

if __name__ == "__main__":
    # Create model instance
    model = CoTNet3D_Model(num_classes=4)
    # Print model structure and number of parameters
    print(model)
    print("Number of parameters: {}".format(get_n_params(model)))
    # Initialize input tensor
    input_tensor = torch.randn((1, 1, 36, 128, 128))  # Batch size of 1, 1 channel, depth 36, height and width 128
    # Perform forward pass
    output = model(input_tensor)
    # Print output shape
    print("Model Output Shape:", output.shape)
