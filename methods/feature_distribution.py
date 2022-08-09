"""
通过简单的特征提取网络获取特征向量
"""
import torch


class FeatureSummary:
    def __init__(self, model_name="mobilenet_v3_small", use_gpu=False):
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.model_path = 'pytorch/vision:v0.12.0'
        self.features_out_hook = []
        self.init_model()
        self.init_feature_layer()

    def hook(self, module, input, output):
        self.features_out_hook.append(output)
        return None
    
    def init_model(self):
        self.model = torch.hub.load(self.model_path, self.model_name, pretrained=True)
        self.model.eval()
        if self.use_gpu:
            self.model.cuda()

    def init_feature_layer(self):
        if self.model_name == 'mobilenet_v3_small':
            self.feature_layer = 'features.12.2'
        else:
            self.feature_layer = 'features.12.2'
        for name, module in self.model.named_modules():
            if name == self.feature_layer:
                self.hadle = module.register_forward_hook(self.hook)
        
    def get_feature(self, dataset):
        torch.cuda.empty_cache()
        dataset = torch.from_numpy(dataset)
        if self.use_gpu:
            dataset = dataset.cuda()
        with torch.no_grad():
            x = self.model(dataset)
        res = self.features_out_hook[0]
        self.features_out_hook = []
        return res
