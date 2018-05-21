from model import EDSR
import scipy.misc
import argparse
import data
import os
import time
from PIL import Image
parser = argparse.ArgumentParser()
parser.add_argument("--dataset",default="data/General-100")
parser.add_argument("--imgsize",default=100,type=int)
parser.add_argument("--scale",default=2,type=int)
parser.add_argument("--layers",default=32,type=int)
parser.add_argument("--featuresize",default=256,type=int)
parser.add_argument("--batchsize",default=10,type=int)
parser.add_argument("--savedir",default="saved_models")
parser.add_argument("--iterations",default=1000,type=int)
parser.add_argument("--numimgs",default=5,type=int)
parser.add_argument("--outdir",default="out")
parser.add_argument("--image")
parser.add_argument("--bicubic")

args = parser.parse_args()
if not os.path.exists(args.outdir):
	os.mkdir(args.outdir)

down_size = args.imgsize//args.scale
network = EDSR(down_size,args.layers,args.featuresize,scale=args.scale)
network.resume(args.savedir)


if args.image:
	start_time = time.time()
	x = scipy.misc.imread(args.image)	
	
	#bicubic = scipy.misc.imread(args.bicubic)
	bicubic = scipy.misc.imresize(x,(x.shape[0]*args.scale,x.shape[1]*args.scale),'bicubic')
	print(x.shape,bicubic.shape)

else:
	print("No image argument given")


inputs = x
outputs = network.predict(x,bicubic)
print('time --> %.4f'%(time.time()-start_time))

#save the result image
if args.image:
	scipy.misc.imsave(args.outdir+"/input_"+'x'+str(args.scale)+'.jpg',inputs)
	scipy.misc.imsave(args.outdir+"/output_"+'x'+str(args.scale)+'.jpg',outputs)

