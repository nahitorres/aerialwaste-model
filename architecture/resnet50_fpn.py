"""
* Upsampling:   nearest neighbor
* Merging:      concatenation

"""

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from architecture import torchutils  # noqa: E402
from architecture import resnet50  # noqa: E402


def get_strided_size(orig_size, stride):
    return ((orig_size[0]-1)//stride+1, (orig_size[1]-1)//stride+1)

def get_strided_up_size(orig_size, stride):
    strided_size = get_strided_size(orig_size, stride)
    return strided_size[0]*stride, strided_size[1]*stride


class Net(nn.Module):
    """FPN with ResNet50 Backbone"""

    def __init__(self, num_classes, pretrained=True, first_trainable=0):
        super(Net, self).__init__()

        self.num_classes = num_classes
        self.first_trainable = first_trainable
        self.resnet50 = resnet50.resnet50(
            # Provare con 2 e 1 nell'ultimo stride
            pretrained=pretrained, strides=(2, 2, 2, 2))

        # First Backbone ResNet Layer
        self.stage0 = nn.Sequential(self.resnet50.conv1,
                                    self.resnet50.bn1,
                                    self.resnet50.relu,
                                    self.resnet50.maxpool)

        # Backbone ResNet Layers (Bottom-up Layers)
        self.stage1 = nn.Sequential(self.resnet50.layer1)
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)

        # Top Layer
        self.toplayer = nn.Conv2d(
            2048, 256, kernel_size=1, stride=1, padding=0)

        # Lateral Layers
        self.latlayer1 = nn.Conv2d(
            1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(
            512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(
            256, 256, kernel_size=1, stride=1, padding=0)

        # Smooth Layers
        self.smooth1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(768, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1)

        # Fully Connected Layer
        self.fc = nn.Linear(256, num_classes)

        # Last Fully Connected
        self.classifier = nn.Linear(4*num_classes, num_classes)

        self.backbone = nn.ModuleList(
            [self.stage0, self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList(
            [self.toplayer, self.latlayer1, self.latlayer2, self.latlayer3,
             self.smooth1, self.smooth2, self.smooth3, self.fc,
             self.classifier])

    def forward(self, x):
        # Bottom-up pathway (ResNet)
        c1 = self.stage0(x)
        c2 = self.stage1(c1)
        c3 = self.stage2(c2).detach()
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)

        # Top-down pathway
        p5 = self.toplayer(c5)
        p4 = self._upsample_cat(p5, self.latlayer1(c4))
        p3 = self._upsample_cat(p4, self.latlayer2(c3))
        p2 = self._upsample_cat(p3, self.latlayer3(c2))

        # Smoothing (de-aliasing effect)
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        # Global average pooling
        p5 = torchutils.gap2d(p5, keepdims=True)
        p4 = torchutils.gap2d(p4, keepdims=True)
        p3 = torchutils.gap2d(p3, keepdims=True)
        p2 = torchutils.gap2d(p2, keepdims=True)

        # Flattening
        p5 = p5.view(p5.size(0), -1)
        p4 = p4.view(p4.size(0), -1)
        p3 = p3.view(p3.size(0), -1)
        p2 = p2.view(p2.size(0), -1)

        # Fully connected layers
        out5 = F.relu(self.fc(p5))
        out4 = F.relu(self.fc(p4))
        out3 = F.relu(self.fc(p3))
        out2 = F.relu(self.fc(p2))

        # Concatenate the predictions (classification results)
        # of each of the pyramid features
        out = torch.cat([out5, out4, out3, out2], dim=1)

        # Last fully connected layer (classifier)
        out = self.classifier(out)

        return out

    def _upsample_cat(self, x, y):
        """Upsample and concatenate two feature maps

        Parameters
        ----------
        x : nn.Variable
            Top feature map to be upsampled
        y : nn.Variable
            Lateral feature map

        Returns
        -------
        nn.Variable
            Concatenated feature map

        Note
        ----
        In PyTorch, when input size is odd, the upsampled feature map with
        nearest upsampling maybe not equal to the lateral feature map size
        (e.g. original input size: [N, _, 15, 15] ->
        conv2d feature map size: [N, _, 8, 8] ->
        upsampled feature map size: [N, _, 16, 16]).
        So we choose bilinear upsample which supports arbitrary output sizes.
        """
        _, _, H, W = y.size()
        upsampled_x = F.interpolate(
            x, size=(H, W), mode="nearest")

        return torch.cat([upsampled_x, y], dim=1)

    def train(self, mode=True):
        pass

    def trainable_parameters(self):
        return (
            list(self.backbone[self.first_trainable:].parameters()),
            list(self.newly_added.parameters()),
        )

class CAM(Net):
    """Produces the final global class activation map (CAM) obtained by
    merging the intermediate cams of the feature pyramid network intermediate
    layers"""

    def __init__(self, num_classes, pretrained=False):
        super(CAM, self).__init__(num_classes, pretrained)

    def forward(self, x):
        image_size = [
            torch.tensor([x.shape[2]]),
            torch.tensor([x.shape[3]]),
        ]

        # Bottom-up pathway (ResNet)
        c1 = self.stage0(x)
        c2 = self.stage1(c1)
        c3 = self.stage2(c2).detach()
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)

        # Top-down pathway
        p5 = self.toplayer(c5)
        p4 = self._upsample_cat(p5, self.latlayer1(c4))
        p3 = self._upsample_cat(p4, self.latlayer2(c3))
        p2 = self._upsample_cat(p3, self.latlayer3(c2))

        # Smoothing (de-aliasing effect)
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        p_list = [p5, p4, p3, p2]

        cams = [self._compute_cam(p, self.classifier.weight[:,index]) for index, p in enumerate(p_list)]
        strided_up_size = get_strided_up_size(image_size, 16)
        unsqueezed_cams = [c.unsqueeze(dim=0) for c in cams]
        interpolated_cams =\
            [F.interpolate(uc, strided_up_size, mode="bilinear",
                           align_corners=False) for uc in unsqueezed_cams]
        concat_cams = torch.cat(interpolated_cams)
        global_cams = torch.sum(concat_cams, dim=0)

        return global_cams

    def _compute_cam(self, p, scale_weight):

        p *= scale_weight
        
        # Multiply the output of the first 4 stages by the weights
        # Note: the weights tensor must be a 4D tensor to perform the
        # convolution with x
        
        p = F.relu(F.conv2d(p, self.fc.weight[:, :, None, None]))
        # Normalize
        p -= torch.min(p)
        p /= torch.max(p)
        
        # Compute class activation maps
        cam = p[0] + p[1].flip(-1)
        cam = torch.where(torch.isnan(cam), torch.zeros_like(cam), cam)
        return cam

class CAM_PRED(Net):
    """Produces the final global class activation map (CAM) obtained by
    merging the intermediate cams of the feature pyramid network intermediate
    layers"""

    def __init__(self, num_classes, pretrained=False):
        super(CAM_PRED, self).__init__(num_classes, pretrained)

    def forward(self, x):
        image_size = [
            torch.tensor([x.shape[2]]),
            torch.tensor([x.shape[3]]),
        ]

        # Bottom-up pathway (ResNet)
        c1 = self.stage0(x)
        c2 = self.stage1(c1)
        c3 = self.stage2(c2).detach()
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)

        # Top-down pathway
        p5 = self.toplayer(c5)
        p4 = self._upsample_cat(p5, self.latlayer1(c4))
        p3 = self._upsample_cat(p4, self.latlayer2(c3))
        p2 = self._upsample_cat(p3, self.latlayer3(c2))

        # Smoothing (de-aliasing effect)
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        p_list = [p5, p4, p3, p2]
        s_list = [self._process_smoothed_p(p.detach().clone()) for p in p_list]
        out = torch.cat(s_list, dim=1)

        # Last fully connected layer (classifier)
        out = self.classifier(out)
        
        cams = [self._compute_cam(p, self.classifier.weight[:,index]) for index, p in enumerate(p_list)]
        strided_up_size = get_strided_up_size(image_size, 16)
        unsqueezed_cams = [c.unsqueeze(dim=0) for c in cams]
        interpolated_cams =\
            [F.interpolate(uc, strided_up_size, mode="bilinear",
                           align_corners=False) for uc in unsqueezed_cams]
        concat_cams = torch.cat(interpolated_cams)
        global_cams = torch.sum(concat_cams, dim=0)

        return global_cams, out

    def _compute_cam(self, p, scale_weight):

        p *= scale_weight
        
        # Multiply the output of the first 4 stages by the weights
        # Note: the weights tensor must be a 4D tensor to perform the
        # convolution with x
        
        p = F.relu(F.conv2d(p, self.fc.weight[:, :, None, None]))
        # Normalize
        p -= torch.min(p)
        p /= torch.max(p)
        
        # Compute class activation maps
        cam = p[0] + p[1].flip(-1)
        cam = torch.where(torch.isnan(cam), torch.zeros_like(cam), cam)
        return cam

    def _process_smoothed_p(self, p):
        """Applying in sequence global average pooling, flattening and a fully
        connected layer to the smoothed intermediate feature maps obtained
        from the concatenation between the lateral layers and the top-down
        layers of the Feature Pyramid Network.

        Parameters
        ----------
        p : [type]
            The intermediate feature maps of the Feature Pyramid Network.

        Returns
        -------
        [type]
            [description]
        """
        gap2d_p = torchutils.gap2d(p, keepdims=True)
        flat_p = gap2d_p.view(gap2d_p.size(0), -1)
        fc_p = F.relu(self.fc(flat_p))

        return fc_p
    
class CAM_SCALES(Net):
    """Produces a list of class activation maps (CAMs), one for each of the
    feature pyramid network intermediate layers"""

    def __init__(self, num_classes, pretrained=False):
        super(CAM_SCALES, self).__init__(num_classes, pretrained)

    def forward(self, x):
        # Bottom-up pathway (ResNet)
        c1 = self.stage0(x)
        c2 = self.stage1(c1)
        c3 = self.stage2(c2).detach()
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)

        # Top-down pathway
        p5 = self.toplayer(c5)
        p4 = self._upsample_cat(p5, self.latlayer1(c4))
        p3 = self._upsample_cat(p4, self.latlayer2(c3))
        p2 = self._upsample_cat(p3, self.latlayer3(c2))

        # Smoothing (de-aliasing effect)
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        # Multiply the output of the first 4 stages by the weights
        # Note: the weights tensor must be a 4D tensor to perform the
        # convolution with x
        p5 = F.relu(F.conv2d(p5, self.fc.weight[:, :, None, None]))
        p4 = F.relu(F.conv2d(p4, self.fc.weight[:, :, None, None]))
        p3 = F.relu(F.conv2d(p3, self.fc.weight[:, :, None, None]))
        p2 = F.relu(F.conv2d(p2, self.fc.weight[:, :, None, None]))

        # Computing cams for each intermediate feature map
        p_list = [p5, p4, p3, p2]
        cams = [self._compute_cam(p) for p in p_list]

        return cams

    def _compute_cam(self, p):
        # Normalize
        p -= torch.min(p)
        p /= torch.max(p)
        # Compute class activation maps
        cam = p[0] + p[1].flip(-1)

        return cam

