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
factorS = 0.5
augParams=[
    ("UnifiedTrain/black", "UnifiedTrain/black/aug", 7000*factorS),
    ("UnifiedTrain/blue", "UnifiedTrain/blue/aug", 12000*factorS),
    ("UnifiedTrain/gray", "UnifiedTrain/gray/aug", 5000*factorS*0),
    ("UnifiedTrain/green","UnifiedTrain/green/aug", 13000*factorS),
    ("UnifiedTrain/red","UnifiedTrain/red/aug", 12000*factorS),
    ("UnifiedTrain/white","UnifiedTrain/white/aug", 0*factorS),
    ("UnifiedTrain/yellow", "UnifiedTrain/yellow/aug", 12000*factorS)
]

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






