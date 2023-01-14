from torchvision import transforms,models
from PIL import Image
from torch import optim
import torch
import get_input_arg as parser
import json
import torch.nn as nn

args=parser.get_predict_parser().parse_args()

with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)

#Loading the checkpoint
device='cuda' if (torch.cuda.is_available() and args.gpu) else 'cpu'

checkpoint=torch.load(args.model_path)
model = eval(f"models.{checkpoint['arch']}")(pretrained=True)
classifier = nn.Sequential(
    nn.Linear(25088, checkpoint['hidden_units']),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(checkpoint['hidden_units'], 102),
    nn.LogSoftmax(dim=1))


model.classifier = classifier


model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
    
def predict(image_path, model, topk):
    im=Image.open(image_path)
    transform=transforms.Compose([transforms.Resize(256),
                          transforms.CenterCrop(224),
                          transforms.ToTensor(),
                          transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    image=transform(im) 
    image=image.to(device)
    log_ps=model.forward(image.view(1,3,224,224))
    ps=torch.exp(log_ps)
    return ps.topk(topk,dim=1)



flower_img_path=args.image_path
with torch.no_grad():
    top_ps,top_classes=predict(flower_img_path, model,args.top_k)
    class_list=[cat_to_name[str(x.item())] for x in top_classes[0]]
    print(class_list)