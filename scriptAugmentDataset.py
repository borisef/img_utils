import augFolderImages
import platform, os, shutil
import numpy as np
try:
    import pwd
except:
    pass

def makeFolder(directory, cleanIfExists):
    """
    Make folder if it doesn't already exist
    :param directory: The folder destination path
    """
    if os.path.exists(directory) and cleanIfExists:
        shutil.rmtree(directory)

    if not os.path.exists(directory):
        os.makedirs(directory)




if platform.system() == "Windows":  # In case of a windows platform - Boris
    dataPrePath = r"e:\\projects\\MB2\\cm\\Data\\"
elif pwd.getpwuid(os.getuid())[0] == 'borisef':  # In case of a linux platform - Boris
    dataPrePath = "/home/borisef/projects/cm/Data/"
elif pwd.getpwuid(os.getuid())[0] == 'koby_a':  # In case of a linux platform - Koby
    dataPrePath = r'/media/koby_a/Data/databases/MagicBox/color_net/DB'

# (Train data folder, Augmented Train data folder, number images)
factorS = 0.2
augParamsTrain=[
    ("Kobi/train_colorDB_without_truncation_CLEAN/black", "UnifiedTrain/black/aug", 1500),#7000*factorS*0.3),
    ("Kobi/train_colorDB_without_truncation_CLEAN/blue", "UnifiedTrain/blue/aug", 5000),#12000*factorS*1),
    ("Kobi/train_colorDB_without_truncation_CLEAN/gray", "UnifiedTrain/gray/aug", 1000),#5000*factorS*0.2),
    ("Kobi/train_colorDB_without_truncation_CLEAN/green","UnifiedTrain/green/aug", 1000),#13000*factorS*0.2),
    ("UnifiedTrain/green", "UnifiedTrain/green/aug", 6000),  # 13000*factorS*0.2),
    ("Kobi/train_colorDB_without_truncation_CLEAN/red","UnifiedTrain/red/aug", 4000),#12000*factorS*0.1),
    ("Kobi/train_colorDB_without_truncation_CLEAN/white","UnifiedTrain/white/aug", 0),#0*factorS),
    #("Kobi/train_colorDB_without_truncation_CLEAN/yellow", "UnifiedTrain/yellow/aug", 500),#12000*factorS*0.2)
    #("UnifiedTrain/yellow", "UnifiedTrain/yellow/aug", 2500),#12000*factorS*0.2)
]

augParamsTest=[
    ("UnifiedTest/blue", "UnifiedTest/blue/aug", 500),
    ("UnifiedTest/green", "UnifiedTest/green/aug", 200),
    ("UnifiedTest/red", "UnifiedTest/red/aug", 100),
    #("UnifiedTest/yellow", "UnifiedTest/yellow/aug", 200)
]
augParams = augParamsTrain

augExtension = "_aug.png"
inputWildcard = "*.png"
augExtWithRandomInt = True
cleanOutExistingFolder = True


for tt in augParams:
    folderIn = os.path.join(dataPrePath, tt[0])
    folderOut =os.path.join(dataPrePath, tt[1])
    sizeOut = int(tt[2])
    makeFolder(folderOut, cleanOutExistingFolder)
    augFolderImages.AugFolderInFoderOut(inputPath=folderIn, outputPath=folderOut,
                                        inputWildcard = inputWildcard,
                                        outputExt = augExtension,
                                        augSeq=augFolderImages.seq4color,
                                        recurs = False,
                                        augExtWithRandomInt = augExtWithRandomInt,
                                        numImagesToGenerate= sizeOut)






