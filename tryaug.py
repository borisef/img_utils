import numpy as np
import os, glob, cv2
import random
import imgaug as ia
import imgaug.augmenters as iaa

class struct():
    pass

seqDistortImage = iaa.Sequential([
    #iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
    #iaa.GaussianBlur(sigma=(10, 13.0)), # blur images with a sigma of 0 to 3.0
    #iaa.JpegCompression(compression=(50, 80))
    #iaa.AdditiveGaussianNoise(scale=(0, 0.2*255))
    #iaa.Multiply((0.5, 2.5)) # brightness
    #iaa.MotionBlur(k=random.choice(range(15)))
    #iaa.GammaContrast((0.5, 2)) # mostly not changing color names

    #edges and ripples
    #iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 1.5)) # makes that ugly black and white lines like in our camera
    #iaa.EdgeDetect(alpha=(0.1, 0.5)) # reminds our camera
    #iaa.Emboss(alpha=(0.1, 0.7), strength=(0.5, 1.5)) # also for edges and color, reminds our camera
    #iaa.PiecewiseAffine(scale=(0.025, 0.075)) # elastic distortion (for color training)
    #iaa.ElasticTransformation(alpha=(0.1, 2.0), sigma=random.choice(np.arange(0.15,1,0.1))) # creates artistic ripples effect
    #iaa.imgcorruptlike.GlassBlur(severity=2) # super slow, like ElasticTransformation
    #iaa.imgcorruptlike.DefocusBlur(severity=random.choice(range(5)))
    iaa.imgcorruptlike.ShotNoise(severity=random.choice(range(5))+1) # may be with median filter
],  random_order=True)


seqChromAbber = iaa.Sequential([
    iaa.WithColorspace(
            to_colorspace="RGB",
            from_colorspace="RGB",
            children=iaa.WithChannels(0, iaa.GaussianBlur(sigma=(0, 1)))
        ),
    iaa.WithColorspace(
            to_colorspace="RGB",
            from_colorspace="RGB",
            children=iaa.WithChannels(0, iaa.Affine(translate_px={"x": (-2,2), "y": (-2,2)}, mode = "edge"))
        ),
    iaa.WithColorspace(
            to_colorspace="RGB",
            from_colorspace="RGB",
            children=iaa.WithChannels(1, iaa.GaussianBlur(sigma=(0, 1)))
        ),
    iaa.WithColorspace(
            to_colorspace="RGB",
            from_colorspace="RGB",
            children=iaa.WithChannels(1, iaa.Affine(translate_px={"x": (-2,2), "y": (-2,2)}, mode = "edge"))
        ),
    iaa.WithColorspace(
            to_colorspace="RGB",
            from_colorspace="RGB",
            children=iaa.WithChannels(2, iaa.GaussianBlur(sigma=(0, 1)))
     ),
    iaa.WithColorspace(
            to_colorspace="RGB",
            from_colorspace="RGB",
            children=iaa.WithChannels(2, iaa.Affine(translate_px={"x": (-2,2), "y": (-2,2)}, mode = "edge"))
        ),

])

seqDistortChannels = iaa.Sequential([
# iaa.WithColorspace(
#         to_colorspace="HSV",
#         from_colorspace="RGB",
#         children=iaa.WithChannels(0,iaa.Add((5, 15))
#     )),
# iaa.WithColorspace(
#         to_colorspace="HSV",
#         from_colorspace="RGB",
#         children=iaa.WithChannels(1,iaa.Add((25, 75))
#     )),
# iaa.WithColorspace(
#         to_colorspace="HSV",
#         from_colorspace="RGB",
#         children=iaa.WithChannels(2,iaa.Add((25, 100))
#     ))
#iaa.MultiplyAndAddToBrightness(mul=(0.5, 2), add=(-30, 30))
iaa.MultiplySaturation((0.5, 3)) # o = gray, 10 = crazy color
], random_order=True)

seq4color = iaa.SomeOf((1, 3),[
    iaa.Crop(px=(0, 7)),
    iaa.JpegCompression(compression=(50, 80)),
    iaa.Multiply((0.5, 2.5)), # brightness
    iaa.GammaContrast((0.5, 2)), # mostly not changing color names
    iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 1.5)), # makes that ugly black and white lines like in our camera
    iaa.PiecewiseAffine(scale=(0.025, 0.075)), # elastic distortion (for color training)
    iaa.ElasticTransformation(alpha=(0.1, 2.0), sigma=random.choice(np.arange(0.15,1,0.1))), # creates artistic ripples effect
    iaa.imgcorruptlike.DefocusBlur(severity=random.choice(range(5))+1),
    iaa.MultiplySaturation((0.5, 3)), # o = gray, 10 = crazy color
    iaa.MultiplyAndAddToBrightness(mul=(0.5, 2), add=(-30, 30)),
    iaa.Sometimes(0.2, seqChromAbber),
    iaa.WithColorspace(
        to_colorspace="HSV",
        from_colorspace="RGB",
        children=iaa.WithChannels(0,iaa.Add((5, 15))
                                  )
    ),
    ], random_order=True)


seq4atr = iaa.Sequential(
    [
    iaa.Sometimes(0.1, seqChromAbber),
    iaa.Sometimes(0.5, iaa.OneOf([
        iaa.JpegCompression(compression=(50, 80)),
        iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 1.5)), # makes that ugly black and white lines like in our camera
        iaa.Emboss(alpha=(0.1, 0.7), strength=(0.5, 1.5)) ,# also for edges and color, reminds our camera
        iaa.ElasticTransformation(alpha=(0.1, 2.0), sigma=random.choice(np.arange(0.15,1,0.1))), # creates artistic ripples effect
        iaa.EdgeDetect(alpha=(0.1, 0.5)) # reminds our camera
    ])),
    iaa.Sometimes(0.5,iaa.OneOf([
        iaa.GaussianBlur(sigma=(10, 13.0)), # blur images with a sigma of 0 to 3.0
        iaa.MotionBlur(k=random.choice(range(10))+3),
        iaa.imgcorruptlike.DefocusBlur(severity=random.choice(range(5))+1)
    ])),

    iaa.Sometimes(0.5,iaa.SomeOf((0, 2),[
        iaa.OneOf([
            iaa.Multiply((0.5, 2.5)), # brightness all channels
            iaa.MultiplyAndAddToBrightness(mul=(0.5, 2), add=(-30, 30))
        ]),
        iaa.GammaContrast((0.5, 2)), # mostly not changing color names
        iaa.MultiplySaturation((0.5, 3)) # o = gray, 10 = crazy color
         ], random_order=True))

    ], random_order=False)




def AugmentImageWithIaa(fullImname, seq, outName):
    img = cv2.imread(fullImname)
    img1 = seq(images=[img])


    return img1[0]


inputPath = "e:/projects/MB2/cppFlowATR/media/filterUCLA/"
inputWildcard = "*.tif"

# inputPath = "e:/projects/MB2/cppFlowATR/media/color/"
# inputWildcard = "*.png"
outputPath ="e:/projects/MB2/img_utils/temp"
outputExt = "tif"

if(os.path.exists(outputPath)==False):
    os.mkdir(outputPath)
params = struct()

for fullImname in glob.glob(os.path.join(inputPath,inputWildcard)):
    seq = seq4color
    img = cv2.imread(fullImname)
    imname = os.path.basename(fullImname)
    outName = os.path.join(outputPath, imname)

    img1 = AugmentImageWithIaa(fullImname, seq, outName)
    if(img1 is not None):
        cv2.imwrite(outName,img1)
    print( fullImname + " ---> " + outName)








# img = cv2.imread("e:/projects/MB2/cppFlowATR/media/filterUCLA/00001000.tif")
#
# img1 = seq(images = [img,img,img])
#
# cv2.imwrite("aaa0.tif",img1[0])
# cv2.imwrite("aaa1.tif",img1[1])
# cv2.imwrite("aaa2.tif",img1[2])