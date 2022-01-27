<div align="center">
  <img height="350" src="/images/pytorch.png"/>
<h1>
  Torchvision Object Detection
</h1>
</div>
  Reponun hazırlanma amacı Torchvision kütüphanesi kullanarak object detection işleminin yapılmasıdır.
### Verinin Test Edilmesi
detect.py içerisinde bulunan aşağıdaki kod bloğu üzerinden;
  -Özel .pth dosyanızın yolunu belirtin.
  -Test edilecek görüntünün yolunu belirtin.
  -İmage Size & threshold değerlerinizi belirtin.
Bunlara ek olarak coco_classes.py dosyası içerisinde sınıf etiketlerinin belirtilmesi gerekmektedir.
```
if __name__ == '__main__':
    # torch model load
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True) # .pth dosyasının orijinal modeli
    model.load_state_dict(torch.load('model.pth')) # özel .pth dosyasınızı burada yükleyin.
    model.eval()
    image = read_image("images/1.jpg", 512)
    detect(image, model, threshold=0.5)
```


