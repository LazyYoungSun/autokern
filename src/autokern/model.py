import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms

class Encoder(nn.Module):
    def __init__(self, weights=ResNet18_Weights.IMAGENET1K_V1):
        super(Encoder, self).__init__()
        resnet = resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        return x

class KerningExpert(nn.Module):
    def __init__(self):
        super(KerningExpert, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        return self.network(x)

def predict_kerning(pairs, font_ttf_path, weights_dir, device="cpu"):
    """
    Берет список пар, загружает ВСЕ 5 моделей фолдов из weights_dir,
    усредняет их предсказания и возвращает словарь {('A', 'V'): округленное_значение}
    """
    encoder = Encoder().to(device)
    encoder.eval()
    
    weight_files = glob.glob(os.path.join(weights_dir, "expert_fold*_450_croped_GELU.pth"))
    if not weight_files:
        raise FileNotFoundError(f"В папке {weights_dir} не найдены файлы весов экспертов!")
        
    experts = []
    for wf in sorted(weight_files):
        expert = KerningExpert().to(device)
        expert.load_state_dict(torch.load(wf, map_location=device))
        expert.eval()
        experts.append(expert)
        
    print(f"Ансамблирование: успешно загружено {len(experts)} моделей фолдов для усреднения.")
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])
    
    from autokern.renderer import rendering_kerning_pair
    predictions = {}
    batch_size = 64
    
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i+batch_size]
            imgs = []
            valid_pairs = []
            
            for pair in batch_pairs:
                img = rendering_kerning_pair(font_ttf_path, pair, crop_to_edge=True)
                imgs.append(transform(img))
                valid_pairs.append(pair)
            
            if not imgs:
                continue
                
            batch_tensor = torch.stack(imgs).to(device)
            
            embeddings = encoder(batch_tensor)
            
            batch_outputs = []
            for expert in experts:
                outputs = expert(embeddings).squeeze(-1).cpu().numpy()
                batch_outputs.append(outputs)
                
            mean_outputs = np.mean(np.array(batch_outputs), axis=0)
            
            for pair, val in zip(valid_pairs, mean_outputs):
                predictions[pair] = int(round(float(val)))
                
    return predictions
