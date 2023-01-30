import io
import torchvision
import torchvision.transforms as tt
import torch
import torch.nn as nn
from PIL import Image
from torchvision.utils import save_image


class Wrapper(nn.Module):
    @classmethod
    def install(cls, model, layer):
        model[layer] = cls(model[layer])
    
    def __init__(self, module):
        super().__init__()
        self.__module, self.x = module, None
    
    def forward(self, x):
        self.x = self.__module(x)
        return self.x
  
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "__module":
                raise AttributeError()
        return getattr(self.__module, name)


class ContentLoss(nn.MSELoss):
    setting = { # key of layer: weight 
        21: 1
    }
    
    def __init__(self, model, img):
        super().__init__()
        self.model, self.img, self.temp = model, img, None
        
    def forward(self, var_img):
        if not self.temp:
            self.model(self.img)
            self.temp = [self.model[i].x for i in self.__class__.setting]
        self.model(var_img)    
        temp2 = [self.model[i].x for i in self.__class__.setting]
        return sum([nn.MSELoss()(self.temp[i].detach(), temp2[i]) * j for i, j in enumerate(self.__class__.setting.values())])


class StyleLoss(nn.MSELoss):
    setting = { # key of layer: weight
        0: 1, 
        5: 0.75,
        10: 0.2,
        19: 0.2, 
        28: 0.2
    }
    
    def __init__(self, model, img):
        super().__init__()
        self.model, self.img, self.temp = model, img, None
        
    def forward(self, var_img):
        if not self.temp:
            self.model(self.img)
            self.temp = [self.gram(self.model[i].x) for i in self.__class__.setting]
        self.model(var_img)    
        temp2 = [self.gram(self.model[i].x) for i in self.__class__.setting]
        return sum([nn.MSELoss()(self.temp[i].detach(), temp2[i]) * j for i, j in enumerate(self.__class__.setting.values())])

    @staticmethod
    def gram(x):
        (bs, ch, h, w) = x.size()
        f = x.view(bs, ch, -1)
        return (f.bmm(f.transpose(1, 2)) / (ch * h * w))


class Loss(nn.MSELoss):
    loss_setting = {
        "style": 1e9,
        "content": 1,
    }
    
    def __init__(self, model, img_content, img_style):
        super().__init__()
        self.content = ContentLoss(model, img_content)
        self.style = StyleLoss(model, img_style)
        
    def forward(self, var_img):
        return (self.__class__.loss_setting["content"] * self.content.forward(var_img) +
                self.__class__.loss_setting["style"] * self.style.forward(var_img))

    def get_layers(self):
        return list(self.content.setting.keys()) + list(self.style.setting.keys())


def FileTo(image, max_size=400, shape=None, device="cpu"):
    image = Image.open(image).convert('RGB')
    size = max_size if  max(image.size) > max_size else max(image.size)
    size = shape if shape is not None else size  
    in_transform = tt.Compose([
        tt.Resize(size),
        tt.ToTensor(),
        tt.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    return in_transform(image)[:3,:,:].unsqueeze(0).to(device)

def ToBytesIO(tensor):
    tensor = tensor.detach().to("cpu").squeeze()
    #tensor = tt.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(tensor)
    b = io.BytesIO()
    save_image(tensor, b, "PNG")
    b.seek(0)
    return b


def process(content_file, style_file):
    content = FileTo(content_file, device=device, max_size=256)
    style = FileTo(style_file, device=device, shape=content.shape[-2:])

    get_loss = Loss(model, content, style)
    for i in get_loss.get_layers():
        Wrapper.install(model, i)

    var = content.clone().requires_grad_(True).to(device)
    opt = torch.optim.Adam([var], 0.003)

    for i in range(500+1):
        opt.zero_grad()    
        loss = get_loss(var)
        loss.backward()
        opt.step()

    output_file = ToBytesIO(var)
    return output_file


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.vgg19(pretrained=True).features.to(device).eval()
