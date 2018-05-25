import data
import argparse
from model import EDSR


parser = argparse.ArgumentParser()
parser.add_argument("--dataset",default="data/General-100")
parser.add_argument("--imgsize",default=100,type=int)
parser.add_argument("--scale",default=2,type=int)
parser.add_argument("--layers",default=32,type=int)
parser.add_argument("--featuresize",default=256,type=int)
parser.add_argument("--batchsize",default=16,type=int)  #10
parser.add_argument("--savedir",default='saved_models')
parser.add_argument("--iterations",default=1000,type=int)
parser.add_argument("--lr",default=0.001,type=float)
parser.add_argument("--scaling_factor",default=0.5, type=float)
parser.add_argument("--load_model",default='',type=str)

args = parser.parse_args()
#get the train data
print('start loading...')
epoch_has_step = data.load_dataset(args.dataset, args.imgsize, args.batchsize)
print('load dataset complit...')

args.imgsize = args.imgsize - (args.imgsize % args.scale)

down_size = args.imgsize//args.scale

network = EDSR(down_size,args.layers,args.featuresize,args.scale, output_channels = 3,sc_factor = args.scaling_factor)


#put the data into the batch ,4 argments
network.set_data_fn(data.get_batch,(args.batchsize,args.imgsize,down_size),data.get_test_set,(args.imgsize,down_size))

network.train(args.iterations, args.savedir, args.lr, args.load_model, epoch_has_step)

