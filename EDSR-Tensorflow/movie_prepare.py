import os
import shutil
import scipy.misc

class Movie_prepare(object):

	def __init__(self,args):
		self.pare ={}		
		self.pare['mv_name'] = args['mv_name']
		self.pare['rate'] = args['rate']
		self.pare['ss'] = args['ss']
		self.pare['t'] = args['t']
		self.pare['b'] = args['b']
		self.pare['outdir'] = args['outdir']
		self.pare['scale'] = args['scale']
		
		if os.path.exists(self.pare['outdir']):
			shutil.rmtree(self.pare['outdir'])
			print('old data has benn Del')
		
		os.mkdir(self.pare['outdir'])
		print('To prepare the movie')

	def get_short_movie(self):		
		os.system('ffmpeg -i %s -r %d -ss %s -t %s %s/%%03d.png'
					%(self.pare['mv_name'],
						self.pare['rate'],
						self.pare['ss'],
						self.pare['t'],
						self.pare['outdir'])
					)
		print('get short movie into the image')


	def get_img_list(self):
		img_list = []
		for filename in os.listdir(self.pare['outdir']):
			if filename.endswith('png'):
				img_list.append(filename)
				#print(filename) 
		img_list.sort(reverse=False)
		
		self.list = img_list
		return img_list	


	def down_sample(self):
		for img in self.list:
			x = scipy.misc.imread(self.pare['outdir']+'/'+img)
			x = scipy.misc.imresize(x,(x.shape[0]//self.pare['scale'],x.shape[1]//self.pare['scale']),'bicubic')
			self.del_img(img)
			scipy.misc.imsave(self.pare['outdir']+'/'+img,x)
		print('the image is down sample')


	def del_img(self,filename):
		os.remove(self.pare['outdir']+'/'+filename)
		print('%s has del'%(filename))


	def make_movie(self):
		name_f,ep = os.path.splitext(self.pare['mv_name'])
		os.system('ffmpeg -r %d -i %s/%%03d.png -r %d -b %s %s/%s'
					%(self.pare['rate'],
					self.pare['outdir'],
					self.pare['rate'],
					self.pare['b'],
					self.pare['outdir'],
					name_f+'_sr'+ep)
				)
		
		
		
