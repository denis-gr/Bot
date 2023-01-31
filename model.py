import io
import torchvision
import torchvision.transforms as tt
import torch
import torch.nn as nn
from PIL import Image
from torchvision.utils import save_image


class Wrapper(nn.Module):
    """Класс прослойка для удобного получения промежуточных результатов
    Просто используйте Wrapper.install на нужном модуле сети и при каждом
    прямом проходе через всю сеть будет обновляться свойство .x этого
    модуля, в котором находятся промежуточные результаты """

    @classmethod
    def install(cls, model, layer):
        """Получает layer - это ключ от model, по которому из model
        можно получить нужный модуль
        Метод класса, создает экземпляр Wrapper, в который передан нужный
        модуль и заменят этот модулю этой оберткой"""
        model[layer] = cls(model[layer])
    
    def __init__(self, module):
        """Получает нужный модуль"""
        super().__init__()
        self.__module, self.x = module, None
    
    def forward(self, *args, **kwargs):
        self.x = self.__module(*args, **kwargs)
        return self.x
  
    def __getattr__(self, name):
        # Пытаемся получить значение из этого объекта
        # Если его нет - получаем из ранее переданного модуля
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.__module, name)


class BaseLoss(nn.MSELoss):
    """Контент/Стиль лосс для этого метода переноса стиля
    get_loss = ContentLoss(model, img, is_use_gram=False)
    loss = get_loss(var)
    model - модель, которая используется для получения признаком
    Нужные модули должны быть обработаны, см класс Wrapper
    img - картинка с нужным контентом
    var - переменная-картинка, которую нужно оптимизировать
    is_use_gram - использовать ли матрицу грамма, используйте 
    False - для контент лосс, True для стиль лосса
    Ключи по котором из model можно получить модуль
    из которого и получает фитчи, а также вес этого
    Нужно указать в переменной класс setting, так что
    ContentLoss.setting[module_key] == weight_this_fitcha
    """
    
    def __init__(self, model, img, is_use_gram=False):
        super().__init__()
        self.model, self.img, self.temp  = model, img, None
        # Функция, что будет использована перед MSELoss над признаками
        self.fun = self.gram if is_use_gram else lambda x: x
        
    def forward(self, var_img):
        # Если этот метод ещё не выполнять, то получаем и сохраняем 
        # Признаки для исходный контент картинки, чтоб ускорить вычисления
        if not self.temp:
            # Делаем проход, чтоб обновить промежуточные результаты
            self.model(self.img) 
            self.temp = { i: self.fun(self.model[i].x.detach())
                                            for i in self.__class__.setting }
        # При каждом вызове получаем нужные признаки для оптимизируемои картинки
        self.model(var_img)    
        temp2 = { i: self.fun(self.model[i].x) for i in self.__class__.setting }
        mce_losses = { i : nn.MSELoss()(self.temp[i], temp2[i]) for i in temp2 }
        # Делаем взвешенную сумму
        summ = sum(mce_losses[i] * j for i, j in self.__class__.setting.items())
        return summ

    @staticmethod
    def gram(x):
        """Возвращает матрицу Грамма из переданной матрицы
        ("обобщенно" для многомерных тензоров)"""
        (bs, ch, h, w) = x.size()
        f = x.view(bs, ch, -1)
        return (f.bmm(f.transpose(1, 2)) / (ch * h * w))


class ContentLoss(BaseLoss):
    setting = { 
        21: 1
    }


class StyleLoss(BaseLoss):
    setting = {
        0: 1, 
        5: 0.75,
        10: 0.2,
        19: 0.2, 
        28: 0.2
    }


class Loss(nn.MSELoss):
    """Итоговый лосс для этого метода
    Имеет атрибут класса dict
    setting["s"] == вес_лосса_стиля
    setting["c"] == вес_лосса_контента
    """
    setting = {
        "s": 1e9,
        "c": 1,
    }
    
    def __init__(self, model, img_content, img_style):
        super().__init__()
        self.c = ContentLoss(model, img_content)
        self.s = StyleLoss(model, img_style, is_use_gram=True)
        
    def forward(self, var_img):
        return (self.__class__.setting["c"] * self.c.forward(var_img) +
                self.__class__.setting["s"] * self.s.forward(var_img))

    def get_layers(self):
        """Возвращает список ключей всех нужных для лосса моделей"""
        return list(self.c.setting.keys()) + list(self.s.setting.keys())


def FileTo(image, max_size=400, shape=None, device="cpu"):
    """Преобразует файл-картинку (image) в тензор,
    получает max_size (максимальный размер, большие будут уменьшены)
    shape - выходной тензор будут иметь этот shape
    device - где должен быть выходной тензор
    """
    image = Image.open(image).convert('RGB')
    size = max_size if max(image.size) > max_size else max(image.size)
    size = shape if shape is not None else size  
    in_transform = tt.Compose([
        tt.Resize(size),
        tt.ToTensor(),
        # Потому что vgg обучалась на нормализованных данных
        tt.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    return in_transform(image)[:3,:,:].unsqueeze(0).to(device)

def ToBytesIO(tensor):
    """Преобразует тензор в BytesIO, у которого интерфейс почти как у файла"""
    tensor = tensor.detach().to("cpu").squeeze()
    # С нормализацией на  выходе- хуже
    #tensor = tt.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(tensor)
    b = io.BytesIO()
    save_image(tensor, b, "PNG")
    b.seek(0)
    return b


def process(content_file, style_file):
    """Получает две картинки и переносит стиль с одной на другую"""
    content = FileTo(content_file, device=device, max_size=256)
    style = FileTo(style_file, device=device, shape=content.shape[-2:])

    get_loss = Loss(model, content, style)
    # Готовим модель для использования
    for i in get_loss.get_layers():
        Wrapper.install(model, i)

    # Получаем переменную для оптимизации размерности, что и картинки
    var = content.clone().requires_grad_(True).to(device)
    opt = torch.optim.Adam([var], 0.003)

    for i in range(2000+1):
        opt.zero_grad()    
        loss = get_loss(var)
        loss.backward()
        opt.step()

    output_file = ToBytesIO(var)
    return output_file


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.vgg19(pretrained=True).features.to(device).eval()
