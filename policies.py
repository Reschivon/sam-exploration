import random
import torch
from torchvision import transforms
import models
from matplotlib import pyplot as plt

class DQNPolicy:
    def __init__(self, cfg, action_space, train=False, random_seed=None):
        self.cfg = cfg
        self.action_space = action_space
        self.train = train

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net = self.build_network()
        self.transform = transforms.ToTensor()

        # Resume from checkpoint if applicable
        if self.cfg.checkpoint_path is not None:
            model_checkpoint = torch.load(self.cfg.model_path, map_location=self.device)
            self.policy_net.load_state_dict(model_checkpoint['state_dict'])
            if self.train:
                self.policy_net.train()
            else:
                self.policy_net.eval()
            print("=> loaded model '{}'".format(self.cfg.model_path))

        if random_seed is not None:
            random.seed(random_seed)

    def build_network(self):
        raise NotImplementedError

    def apply_transform(self, s):
        return self.transform(s).unsqueeze(0)

    def step(self, state, exploration_eps=None, debug=False, mask_invalid=False):
        if exploration_eps is None:
            exploration_eps = self.cfg.final_exploration
        state = self.apply_transform(state).to(self.device)
        with torch.no_grad():
            output = self.policy_net(state).squeeze(0)
            plt.imshow(output.detach().cpu().squeeze())
        if random.random() < exploration_eps:
            action = random.randrange(self.action_space)
        else:
            if mask_invalid:
                # Make 0 values of SPFA channel (obstacles, out of bounds) ineligible for action selection
                output[state[:, 3] != 0] = -99999
            action = output.view(1, -1).max(1)[1].item()
        info = {}
        if debug:
            info['output'] = output.squeeze(0)
        return action, info

class SteeringCommandsPolicy(DQNPolicy):
    def build_network(self):
        return torch.nn.DataParallel(
            models.SteeringCommandsDQN(num_input_channels=self.cfg.num_input_channels, num_output_channels=self.action_space) \
        ).to(self.device)

class DenseActionSpacePolicy(DQNPolicy):
    def build_network(self):
        return torch.nn.DataParallel(
            models.DenseActionSpaceDQN(num_input_channels=self.cfg.num_input_channels) \
        ).to(self.device)

class DeepLabPolicy(DQNPolicy):
    def build_network(self):
       return models.deeplabv3_resnet18(num_input_channels=self.cfg.num_input_channels).to(self.device)

        # return models.DenseActionSpaceDQN(num_input_channels=self.cfg.num_input_channels).to(self.device)


from ConvNeXt.semantic_segmentation.backbone.convnext import ConvNeXt
from mmseg.models.decode_heads.uper_head import UPerHead
class Stupid(torch.nn.Module):
    def __init__(self, num_input_channels=3, num_output_channels=1):
        super().__init__()
        self.backbone = ConvNeXt(
                in_chans=3,
                depths=[3, 3, 5, 3], 
                dims=[96, 192, 384, 768], 
                drop_path_rate=0.4,
                layer_scale_init_value=1.0,
                out_indices=[0, 1, 2, 3],
            )
        self.decode_head=UPerHead(
                # in_channels=[96, 192, 384, 768],
                in_channels=[64, 128, 256, 512],
                num_classes=1,
                in_index=[0, 1, 2, 3],
                pool_scales=(1, 2, 3, 6),
                # channels=512,
                channels=256,
                dropout_ratio=0.1,
                norm_cfg=None, # dict(type='BN', requires_grad=True),
                align_corners=False,
                act_cfg=dict(type='ReLU', inplace=False)
            )
        
        self.up = torch.nn.Upsample(scale_factor=4, mode='nearest')

    def forward(self, x):
        x = self.backbone(x)
        x = self.decode_head(x)
        # x = x[:, 0, :, :] / x[:, 1, :, :]

        # add dim so upscaling works
        # x = x[:, None, :, :]
        x = self.up(x)
        x = x.squeeze(1)
        # plt.imshow(x.cpu().detach().squeeze())
        # plt.show()
        return x

class ConvNextPolicy(DQNPolicy):
    def build_network(self):
        return Stupid() \
        .to(self.device)
