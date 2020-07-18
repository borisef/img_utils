import numpy as np
import os, glob, cv2
import random
import pandas as pd


#GLOBALS
pp = []
zz = []
gg = []

extraParams = {"preserve_aspect_ratio": False, "alpha": 0.8}
table_columns = ["image", "x1", "y1", "x2", "y2", "class", "score"]


def createFolderIfNeed(foldername):
	if(not os.path.exists(foldername)):
		os.mkdir(foldername)


def resizeTo(im, w):
	rsz = w/im.shape[1]
	rr = int(im.shape[0]*rsz)
	cc = int(im.shape[1]*rsz)
	im1 = cv2.resize(im,(cc, rr), interpolation= cv2.INTER_LANCZOS4)
	return im1, rsz

def DrawTarget2Rectangle(im,rect,targetsFolderOrImg = None, extraParamsDict = None ):
	if(os.path.isdir(targetsFolderOrImg)):
		imgs = glob.glob(targetsFolderOrImg + '/*.jpg')
		targetImg = random.choice(imgs)
	else:
		targetImg = targetsFolderOrImg



	tim = cv2.imread(targetImg)
	cc = rect[1][0] - rect[0][0]
	rr = rect[1][1] - rect[0][1]

	if((extraParams is not None) and (extraParams["preserve_aspect_ratio"])):
		rr = int(tim.shape[0]*cc/tim.shape[1])

	restim = cv2.resize(tim, (cc, rr), interpolation=cv2.INTER_LANCZOS4)
	alpha = 1
	if ((extraParams is not None) and (extraParams["alpha"])):
		alpha = extraParams["alpha"]
	im[rect[0][1]:(rect[0][1]+rr), rect[0][0]:(rect[0][0]+cc)] = restim*alpha + im[rect[0][1]:(rect[0][1]+rr), rect[0][0]:(rect[0][0]+cc)]*(1 - alpha)
	return (im, targetImg)



def click_and_keep(event, x, y, flags, param):
	# grab references to the global variables
	global refPt
	global gg
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]

	# performed
	if event == cv2.EVENT_LBUTTONUP:
		refPt.append((x, y))
		gg.append(refPt)
		L = len(gg)
		print(L)

def whileTrueWindow(nameWind,im,clone_im, fh = None, fhParams = None):
	global gg
	gg = []
	global targetsFolder
	cv2.namedWindow(nameWind)
	cv2.setMouseCallback(nameWind, click_and_keep)
	lastL = 0
	listImgs =[]
	while True:
		# display the image and wait for a keypress
		cv2.imshow(nameWind, im)
		key = cv2.waitKey(1) & 0xFF
		# if the 'r' key is pressed, reset the cropping region
		if key == ord("r"):
			im = clone_im.copy()
			gg = []
			listImgs = []
		# if the 'c' key is pressed, break from the loop
		elif key == ord("c"):
			break

		if key == ord("d"):
			im = clone_im.copy()
			if (len(gg)):
				gg.pop()
				if (len(listImgs)):
					listImgs.pop()
		# draw a rectangle around the region of interest
		if(len(gg) != lastL):

			for i, p in enumerate(gg):
				if(fh is not None):
					if((lastL < len(gg) and i >= lastL)):
						im, tim = fh(im,p, targetsFolder, fhParams)
						listImgs.append(tim)
					elif(lastL > len(gg)):
						im, tim = fh(im, p, listImgs[i], fhParams)
				else:
					cv2.rectangle(im, p[0], p[1], (0, 255, 0), 2)
			lastL = len(gg)

	cv2.destroyWindow(nameWind)
	return (gg, listImgs)



def InfuseInSingleImage(inputName, outputFolder, infuseParams = None):
	imgName = inputName
	orig_image = cv2.imread(imgName)  # BGR
	imsmall, rsz = resizeTo(orig_image, 1000)
	clone_small = imsmall.copy()
	clone_orig = orig_image.copy()
	outDF = pd.DataFrame([], columns=table_columns)

	#
	zz = whileTrueWindow("Select large regions for targets use c(ontinue),d(elete last),r(estart)", imsmall, clone_small)[0]
	zz_orig = (np.array(zz) / rsz).astype(int)

	for rp_orig in zz_orig:  # for each subwindow
		# rp_orig = (np.array(rp)/rsz).astype(int)
		roi_orig = orig_image[rp_orig[0][1]:rp_orig[1][1], rp_orig[0][0]:rp_orig[1][0]]
		roi_small, rsz1 = resizeTo(roi_orig, 1000)
		clone_roi_small = roi_small.copy()
		pp, list_imgs = whileTrueWindow("Select object locations use c(ontinue),d(elete last),r(estart)", roi_small, clone_roi_small, DrawTarget2Rectangle, extraParams)
		if(len(pp)==0):
			continue
		# correct pp to orig_image
		pp_orig = (np.array(pp) / (rsz1)).astype(int)  # to roi_orig (size)
		pp_orig[:, :, 0] = pp_orig[:, :, 0] + rp_orig[0][0]  # to image_orig (offset)
		pp_orig[:, :, 1] = pp_orig[:, :, 1] + rp_orig[0][1]
		for i, p in enumerate(pp_orig):
			# cv2.rectangle(clone_orig, (p[0,0],p[0,1]), (p[1,0],p[1,1]), (0, 255, 0), 2)
			clone_orig = DrawTarget2Rectangle(clone_orig, p, list_imgs[i], infuseParams)[0]  #
			# add rows to DF
			nn = os.path.basename(imgName)
			dd = {"image": nn, "x1": p[0, 0], "y1": p[0, 1], "x2": p[1, 0], "y2": p[0, 1], "class": "margema",
				  "score": 1.0}
			outDF = outDF.append(dd, ignore_index=True)

	cv2.imwrite("results.png", clone_orig)
	# image save to output folder
	createFolderIfNeed(outputFolder)
	imname_only = os.path.basename(imgName)
	cv2.imwrite(os.path.join(outputFolder, imname_only), clone_orig)

	print(pp)
	print(pp_orig)
	return outDF, clone_orig




#====================================

if __name__ == "__main__":

	targetsFolder = "targets"
	backgroundsFolder = "backgrounds"
	outputFolder = "outputInfuseTargets"

	outDF = pd.DataFrame([], columns=table_columns)

	bckg_imgs = glob.glob(backgroundsFolder + '/*.tif')

	for imgName in bckg_imgs:
		outDF1, outImg = InfuseInSingleImage(imgName, outputFolder)
		outDF=outDF.append(outDF1)

	#save table
	txtName = os.path.join(outputFolder, "labels.csv")
	outDF.to_csv(txtName, header=True, index=False)
	print("OK")

