import data
import argparse
from model import EDSR
parser = argparse.ArgumentParser()
parser.add_argument("--dataset",default="../data/General-100")
parser.add_argument("--imgsize",default=100,type=int)
parser.add_argument("--scale",default=3,type=int)
parser.add_argument("--layers",default=32,type=int)
parser.add_argument("--featuresize",default=256,type=int)
parser.add_argument("--batchsize",default=32,type=int)
parser.add_argument("--savedir",default='saved_models')
parser.add_argument("--iterations",default=35,type=int)
parser.add_argument("--load_model",default='',type=str)
args = parser.parse_args()

args.imgsize = args.imgsize - (args.imgsize % args.scale)

each_epoch_step = data.load_dataset(args.dataset,args.imgsize,args.batchsize)
down_size = args.imgsize//args.scale

network = EDSR(down_size,args.layers,args.featuresize,args.scale)

network.set_data_fn(data.get_batch,(args.batchsize,args.imgsize,down_size),data.get_test_set,(args.imgsize,down_size))

network.train(args.iterations,args.savedir,args.load_model,each_epoch_step)
