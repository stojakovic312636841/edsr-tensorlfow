import data
import argparse
from model import EDSR
parser = argparse.ArgumentParser()
parser.add_argument("--dataset",default="../data/General-100")
parser.add_argument("--imgsize",default=100,type=int)
parser.add_argument("--scale",default=4,type=int)
parser.add_argument("--layers",default=32,type=int)
parser.add_argument("--featuresize",default=256,type=int)
parser.add_argument("--batchsize",default=32,type=int)
parser.add_argument("--savedir",default='saved_models')
parser.add_argument("--iterations",default=35,type=int)
parser.add_argument("--load_model",default='',type=str)
parser.add_argument("--mult_gpu", action = 'store_true', help='the train use mult gpu')
args = parser.parse_args()

#use the tf queue to load img with less time
USE_QUEUE_LOADING = False
if args.mult_gpu == True:
	USE_QUEUE_LOADING = True



args.imgsize = args.imgsize - (args.imgsize % args.scale)
print('start to load the train data...')
each_epoch_step = data.load_dataset(args.dataset,args.imgsize,args.batchsize)		
	
down_size = args.imgsize//args.scale

network = EDSR( img_size = down_size,
				num_layers = args.layers,
				feature_size = args.featuresize,
				scale = args.scale, 
				output_channels = 3,
				batch_size = args.batchsize, 
				step_in_epo = each_epoch_step, 
				use_mult_gpu = args.mult_gpu,
				is_test = False,
				use_queue = True)


if USE_QUEUE_LOADING is True and args.mult_gpu is True:
	network.load_dataset_queue(args)
	network.set_test_data_fn(data.get_test_set,(args.imgsize,down_size))
else:
	network.set_data_fn(data.get_batch,(args.batchsize,args.imgsize,down_size),data.get_test_set,(args.imgsize,down_size))

network.train(args.iterations,args.savedir,args.load_model,each_epoch_step)

