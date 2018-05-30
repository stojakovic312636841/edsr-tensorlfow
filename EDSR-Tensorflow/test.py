from model import EDSR
import scipy.misc
import argparse
import data
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",default="data/General-100")
parser.add_argument("--imgsize",default=100,type=int)
parser.add_argument("--scale",default=4,type=int)
parser.add_argument("--layers",default=32,type=int)
parser.add_argument("--featuresize",default=256,type=int)
parser.add_argument("--batchsize",default=10,type=int)
parser.add_argument("--savedir",default="saved_models")
parser.add_argument("--iterations",default=1000,type=int)
parser.add_argument("--numimgs",default=5,type=int)
parser.add_argument("--outdir",default="out")
parser.add_argument("--image")
parser.add_argument("--mult_gpu", action = 'store_true', help='the test use mult gpu')


args = parser.parse_args()
if not os.path.exists(args.outdir):
	os.mkdir(args.outdir)

args.imgsize = args.imgsize - (args.imgsize % args.scale)

down_size = args.imgsize//args.scale


network = EDSR(down_size,args.layers,args.featuresize,scale=args.scale,use_mult_gpu = args.mult_gpu, is_test = args.mult_gpu, use_queue = False)

network.resume(args.savedir)

if args.image:
	x = scipy.misc.imread(args.image)
else:
	print("No image argument given")


inputs = x
start_time = time.time()
outputs = network.predict(x)

print('cost time  = %.5f'%(time.time() - start_time))

#save the result image
if args.image:
	scipy.misc.imsave(args.outdir+"/input_"+'x'+str(args.scale)+'.png',inputs)
	scipy.misc.imsave(args.outdir+"/output_"+'x'+str(args.scale)+'.png',outputs)
