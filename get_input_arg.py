import argparse

def get_train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir',type=str)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--hidden_units',type=int,help='Choose number of hidden units',default=256)
    parser.add_argument('--arch',type=str,help='Choose architecture',default="vgg11")
    parser.add_argument('--save_dir',type=str,help='Choose directory to save model to',default='flowermodel.pth')
    parser.add_argument('--learning_rate',type=float,help='Chooses Learning rate',default=0.003)
    parser.add_argument('--epochs',type=int,help='Choose epochs',default=1)
    print(parser.parse_args())
    return parser

def get_predict_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path',type=str)
    parser.add_argument('model_path',type=str)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--top_k',type=int,help='Gives top k class names',default=5)
    parser.add_argument('--category_names',type=str,help='Choose file to map class names',default='cat_to_name.json')
    return parser
    