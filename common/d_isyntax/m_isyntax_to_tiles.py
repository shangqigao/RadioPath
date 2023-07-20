###########################
## m_isyntax_to_tiles.py ##
###########################

## goal: format whole slide images in svs format for MIL analysis 

## author: Willem Bonnaffe (w.bonnaffe@gmail.com)

##############
## INITIATE ##
##############

## imports
import os
import numpy as np
import cv2

## import local modules
if __name__ == "__main__":

    from m_wsi_reader import *

else:

    from .m_wsi_reader import *

#
###

###############
## FUNCTIONS ##
###############

## embedd
## goal: insert image array in a black canvas of specified size
## img        - np.array - image to frame
## canvasSize - int      - size in pixels of array in which to insert the image
def embedd(img,canvasSize):
    if (canvasSize - img.shape[0] > 0):
        img = np.concatenate((img,np.zeros((canvasSize - img.shape[0],img.shape[1],img.shape[2]))),axis=0)
    if (canvasSize - img.shape[1] > 0):
        img = np.concatenate((img,np.zeros((img.shape[0],canvasSize - img.shape[1],img.shape[2]))),axis=1)
    return img

## tile
## goal: split image into tiles
## img       - np.array - iamge to tile
## n_tiles   - int      - number of tiles along the width of the image
## tile_size - int      - size of each tile in pixels
def tile(img,n_tiles,tile_size):
    tile_stack = np.zeros((n_tiles,n_tiles,tile_size,tile_size,img.shape[2]))
    for i in range(0,n_tiles):
        for j in range(0,n_tiles):
            tile_stack[i,j] = img[(i*tile_size):((i+1)*tile_size),(j*tile_size):((j+1)*tile_size)]
    return tile_stack

## isyntax_to_tiles_dataset
## goal: divide isyntax image into tiles
## pt_input_folder - string - path to input folder containing slide images in svs format
## pt_dataset      - string - path to output folder
## tile_size       - int    - number of pixels along width of tiles
## n_tiles         - int    - number of tiles along the width of a chunk
## level           - int    - magnification level of slide
def isyntax_to_tiles_dataset(pt_input_folder, pt_dataset, tile_size, n_tiles, level):
    
    ## get list of slide image files
    slideFileNames = os.listdir(pt_input_folder)

    ## create dataset folder 
    if os.path.exists(pt_dataset) == False:
        os.mkdir(pt_dataset)
    
    ## for each slide image file
    for slideFileName in slideFileNames:
        
        ## update
        print("opening " + slideFileName)
        slideID = slideFileName.replace(".isyntax","")
        print("slide ID: " + slideID)
        
        ## check if slide already formatted
        if os.path.exists(pt_dataset + slideID) == False:
    
            ## open slide (openslide) 
            # slide = ops.OpenSlide(pt_input_folder + slideID + ".svs")
            #
            ## open slide (isyntax)
            reader = get_reader_impl(pt_input_folder + slideID + ".isyntax")
            slide = reader(pt_input_folder + slideID + ".isyntax")
             
            ## slide properties 
            slideDim = (np.array(slide.level_dimensions[level])).astype("int")
    
            ## chunk properties
            chunkSize = n_tiles*tile_size
            n_chunks   = (np.ceil(slideDim/chunkSize)).astype("int")
            print("chunk size: " + str(chunkSize))
            print("chunk number: " + str(n_chunks))
    
            ## load slide in chunks
            c = 0
            for i in range(0,n_chunks[0]):
                for j in range(0,n_chunks[1]):
    
                    ## update
                    print("loading chunk " + str(c+1) + "/" + str(n_chunks[0]*n_chunks[1]))
    
                    ## create chunk folder
                    chunk_label = "__C" + str(c) + "_I" + str(i) + "_J" + str(j)
                    if os.path.exists(pt_dataset + slideID + chunk_label + "/") == False: 
                        os.mkdir(pt_dataset + slideID + chunk_label + "/")

                    ## create tile folder
                    if os.path.exists(pt_dataset + slideID + chunk_label + "/" + "tiles/") == False: 
                        os.mkdir(pt_dataset + slideID + chunk_label + "/" + "tiles/")

                    ## 4 image shifts to overlap tiles
                    shift_label = ["SE","SW","NE","NW"]
                    shift_iterator = 0
                    for shift_i in range(0,2): # W <-> E
                        for shift_j in range(0,2): # N <-> S
    
                            ## read region (openslide)
                            # img = np.array(slide.read_region((i*chunkSize,j*chunkSize),level,(chunkSize,chunkSize)))[0:chunkSize,0:chunkSize]
                            #
                            ## read region (isyntax)
                            img = np.array(slide.read_region((i*chunkSize,j*chunkSize),level,(chunkSize,chunkSize),normalize=False)[0])[0:chunkSize,0:chunkSize]
                
                            ## convert colour 
                            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
                            ## shift image
                            img = img[int(shift_i*0.5*tile_size):,int(shift_j*0.5*tile_size):]
                            img = embedd(img, chunkSize)
    
                            ## create tile stack
                            tile_stack = tile(img,n_tiles,tile_size)
    
                            ## select tiles with tissue
                            thumbnail = cv2.resize(img, (n_tiles,n_tiles), interpolation = cv2.INTER_AREA)
                            selected  = np.argwhere((np.std(thumbnail,2) > 5) & (np.mean(thumbnail,2)>50))
                            #
                            # thumbnail = cv2.resize(img, (n_tiles,n_tiles), interpolation = cv2.INTER_AREA)
                            # selected  = np.argwhere((np.mean(thumbnail,2)>0))
    
                            ## write tiles
                            k = 0
                            for s in selected:
                                cv2.imwrite(pt_dataset + slideID + chunk_label + "/" + "tiles/" + "C" + str(c) + "_I" + str(i) + "_J" + str(j) + "_T" + str(k) + "_I" + str(s[0]) + "_J" + str(s[1]) + "_" + str(shift_label[shift_iterator]) + ".png",tile_stack[s[0],s[1]])
                                k = k + 1
    
                            ## iterator
                            shift_iterator = shift_iterator + 1
    
                    ## iterator
                    c = c + 1
    
            ## update
            print("input slide: " + pt_input_folder + slideID + ".svs")
            print("tiles stored at: " + pt_dataset + slideID + "/")
            print("\n")

#
###

##########
## MAIN ##
##########

if __name__ == "__main__":

    ## imports
    import argparse

    ## argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--pt_input_folder', default="/well/rittscher/users/cwy906/workd/datasets/PROMPT/raw/images/isyntax/")
    parser.add_argument('--pt_dataset', default="/well/rittscher/users/cwy906/workd/datasets/PROMPT/formatted/PROMPT_L0_GlandSeg/")
    parser.add_argument('--tile_size', default=1024, type=int)
    parser.add_argument('--n_tiles', default=16, type=int)
    parser.add_argument('--level', default=1, type=int)
    args = parser.parse_args()

    ## tile svs images
    isyntax_to_tiles_dataset(pt_input_folder = args.pt_input_folder, 
            pt_dataset = args.pt_dataset,
            tile_size = args.tile_size,
            n_tiles = args.n_tiles,
            level = args.level)
    

#
###