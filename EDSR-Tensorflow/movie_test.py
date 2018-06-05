from movie_model import EDSR
import scipy.misc
import argparse
import data
import os
import time
from PIL import Image
from movie_prepare import Movie_prepare


parser = argparse.ArgumentParser()
parser.add_argument("--dataset",default="data/General-100")
parser.add_argument("--imgsize",default=100,type=int)
parser.add_argument("--scale",default=4,type=int)
parser.add_argument("--layers",default=32,type=int)
parser.add_argument("--featuresize",default=256,type=int)
parser.add_argument("--batchsize",default=1,type=int)
parser.add_argument("--load_model",default=".")
parser.add_argument("--iterations",default=1000,type=int)
parser.add_argument("--numimgs",default=5,type=int)

parser.add_argument("--outdir",default="movie_out")
parser.add_argument("--image")
parser.add_argument("--video")
parser.add_argument("--rate",default=24,type=int,help='the frame rate in second')
parser.add_argument("--b",default='9000K',type=str,help='the frame braud')
parser.add_argument("--ss",default="00:00:00",type=str,help='the start time point in the movie to clip')
parser.add_argument("--t",default="00:00:00",type=str,help='the continue time point in the movie to clip in the end')
parser.add_argument("--mult_gpu", action = 'store_true', help='the test use mult gpu')


args = parser.parse_args()
#if not os.path.exists(args.outdir):
#	os.mkdir(args.outdir)

if args.video:
	name = os.path.basename(args.video)
	#print(name)
	movie_para = {}
	movie_para['mv_name'] = name
	movie_para['rate'] = args.rate
	movie_para['ss'] = args.ss
	movie_para['t'] = args.t
	movie_para['b'] = args.b
	movie_para['outdir'] = args.outdir
	movie_para['scale'] = args.scale

	mv = Movie_prepare(movie_para)
	mv.get_short_movie()
	img_list = mv.get_img_list()

	#down sample
	mv.down_sample()

else:
	print('NO Video')
	os._exit(0)



t_strat_time = time.time()

image_list = []
for img in img_list:	
	name_f,ep = os.path.splitext(img)
	print(name_f)
	x = scipy.misc.imread(args.outdir+'/'+img)	
	image_list.append(x)

	args.imgsize = min(x.shape[0],x.shape[1]) * args.scale
	args.imgsize = args.imgsize - (args.imgsize % args.scale)
	down_size = args.imgsize//args.scale

network = EDSR(down_size,args.layers,args.featuresize,scale=args.scale,use_mult_gpu = args.mult_gpu, is_test = args.mult_gpu, use_queue = False)

if args.load_model == '.':
	print('the model path is error')
	os._exit(0)

#resume
network.resume(args.load_model)

start_time = time.time()
outputs = network.movie_predict(image_list)

print('cost time  = %.5f'%(time.time() - start_time))


#
for filename in os.listdir(args.outdir):
	if filename.endswith('.png') is True:
		mv.del_img(filename)

i=0
for x in outputs:	
	scipy.misc.imsave(args.outdir+'/'+'%05d'%(i)+'.png',x)
	i=i+1
	print('%d has saved'%(i))



#put the image into movie
mv.make_movie()

for filename in os.listdir(args.outdir):
	if filename.endswith('.png') :
		mv.del_img(filename)

print('total cost time  = %.5f'%(time.time() - t_strat_time))
