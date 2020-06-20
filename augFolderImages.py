import numpy as np
import os, glob, cv2
import random
import imgaug as ia
import imgaug.augmenters as iaa


class struct():
    pass

def AugmentImageWithIaa(fullImname, seq, outName):
    img = cv2.imread(fullImname)
    img1 = seq(images=[img])
    return img1[0]

def folders_in(path_to_parent):
    res = []
    for fname in os.listdir(path_to_parent):
        if os.path.isdir(os.path.join(path_to_parent,fname)):
            res.append(fname)
    return res

def AugFolderInFoderOut(inputPath, inputWildcard,outputPath, outputExt,  augSeq, recurs, numImagesToGenerate,augExtWithRandomInt = False):
    if (os.path.exists(outputPath) == False):
        os.mkdir(outputPath)

    numImagesInFolder = len(glob.glob(os.path.join(inputPath, inputWildcard)))
    numImagesInOutFolder = len(glob.glob(os.path.join(outputPath, "*" + outputExt)))
    numRuns = 1
    if(numImagesToGenerate > 0  and numImagesToGenerate > numImagesInFolder ):
        print("numImagesToGenerate > numImagesInFolder, setting augExtWithRandomInt = True")
        augExtWithRandomInt = True

    if (numImagesToGenerate == 0):  # skip
        return 0
    elif(numImagesToGenerate < 0 or numImagesToGenerate == numImagesInFolder): # exact
        listImages2generate = glob.glob(os.path.join(inputPath, inputWildcard))
    elif (numImagesToGenerate <numImagesInFolder): #subset
        listImages2generate = random.sample(glob.glob(os.path.join(inputPath, inputWildcard)),k = numImagesToGenerate)
    else: # superset
        listImages2generate = random.choices(glob.glob(os.path.join(inputPath, inputWildcard)),k = numImagesToGenerate)

    for fullImname in listImages2generate:
        try:
            seq = augSeq
            imname = os.path.basename(fullImname)
            noextName = os.path.splitext(imname)[0]
            outName = os.path.join(outputPath, noextName + outputExt)
            if(augExtWithRandomInt):
                randIntStr =str(np.random.randint(0,100000)+1000000)[1:]
                outName = os.path.join(outputPath, noextName + "_" + randIntStr + "_" + outputExt)

            img1 = AugmentImageWithIaa(fullImname, seq, outName)
            if (img1 is not None):
                cv2.imwrite(outName, img1)
            print(fullImname + " ---> " + outName)
        except:
            print(fullImname + " ---> FAILED !!!")






    if(recurs):
        ff = folders_in(inputPath)
        for fldr in ff:
            AugFolderInFoderOut(os.path.join(inputPath,fldr), inputWildcard,
                                os.path.join(outputPath,fldr),outputExt, augSeq, recurs)

    return 0



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

seq4color = iaa.SomeOf((1, 3),[
    iaa.Crop(px=(0, 3)),
    iaa.JpegCompression(compression=(40, 60)),
    iaa.OneOf([
        #iaa.Multiply((0.5, 2)),  # brightness
        iaa.GammaContrast((0.5, 2)),  # mostly not changing color names
        iaa.MultiplySaturation((0.5, 2.5)),  # 0 = gray, 10 = crazy color
        iaa.MultiplyAndAddToBrightness(mul=(0.6, 1.7), add=(-20, 20)),
        iaa.WithColorspace(
                to_colorspace="HSV",
                from_colorspace="RGB",
                children=iaa.WithChannels(0,iaa.Add((5, 15))
                                          )
            ),
    ]),

    iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 1.5)), # makes that ugly black and white lines like in our camera
    iaa.PiecewiseAffine(scale=(0.025, 0.075)), # elastic distortion (for color training)
    iaa.ElasticTransformation(alpha=(0.1, 2.0), sigma=random.choice(np.arange(0.15,1,0.1))), # creates artistic ripples effect
    iaa.imgcorruptlike.DefocusBlur(severity=random.choice(range(5))+1),

    iaa.Sometimes(0.2, seqChromAbber),

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

    iaa.Sometimes(0.9,iaa.SomeOf((0, 1),[
        iaa.OneOf([
            iaa.Multiply((0.5, 1.75)), # brightness all channels
            iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-20, 10))
        ]),
        iaa.GammaContrast((0.5, 1.5)), # mostly not changing color names
        iaa.MultiplySaturation((0.5, 2.5)) # o = gray, 10 = crazy color
         ], random_order=True))

    ], random_order=False)


if __name__ == "__main__":

    inputPath = "f:/cm/Data/UnifiedTest/yellow/"
    #inputPath = "/home/borisef/projects/cppFlowATR/media/spliced/"
    inputWildcard = "*.png"

    # inputPath = "e:/projects/MB2/cppFlowATR/media/color/"
    # inputWildcard = "*.png"
    #outputPath ="/home/borisef/temp/temp1"
    outputPath = "e:/projects/MB2/img_utils/temp1/"
    outputExt = "_aug.png"
    recurs = False
    generatePerFolder = -1
    augExtWithRandomInt = False
    AugFolderInFoderOut(inputPath, inputWildcard,outputPath,outputExt, seq4color,recurs, generatePerFolder,augExtWithRandomInt)
