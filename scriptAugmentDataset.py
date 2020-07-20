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
    dataPrePath = "/home/borisef/projects/img_utils/"
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

augParamsKhaki = [
    ('khaki/test/', 'khaki/test/aug', 100),
    ('khaki/train/', 'khaki/train/aug', 2000)
]


augParamsExam1 = [
                    ("Exam1_train/ykhaki","Exam1_train/ykhaki_aug", 100),
                    ("Exam1_train/white","Exam1_train/white_aug", 500),
                    ("Exam1_train/red", "Exam1_train/red_aug", 600),
                    ("Exam1_train/green", "Exam1_train/green_aug", 200),
                    ("Exam1_train/gray","Exam1_train/gray_aug", 1000),
                    ("Exam1_train/blue", "Exam1_train/blue_aug", 200),
                    ("Exam1_train/black", "Exam1_train/black_aug", 500)
                    ]

augParamsMargema = [("crops","crops/aug", 300)]

augParams = augParamsMargema

augExtension = "_aug.png"
inputWildcard = "*.png"
augExtWithRandomInt = True
cleanOutExistingFolder = True
#augSeq = augFolderImages.seq4color
augSeq = augFolderImages.seq4margema

for tt in augParams:
    folderIn = os.path.join(dataPrePath, tt[0])
    folderOut =os.path.join(dataPrePath, tt[1])
    sizeOut = int(tt[2])
    makeFolder(folderOut, cleanOutExistingFolder)
    augFolderImages.AugFolderInFoderOut(inputPath=folderIn, outputPath=folderOut,
                                        inputWildcard = inputWildcard,
                                        outputExt = augExtension,
                                        augSeq=augSeq,
                                        recurs = False,
                                        augExtWithRandomInt = augExtWithRandomInt,
                                        numImagesToGenerate= sizeOut)






