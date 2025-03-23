import shutil
import os
import cv2
import numpy as np
import pandas as pd
import random
def split_train_val(folder_path):
    val_folder = '/home/long/longdata/chúa phù hộ người tên khải/ml med/data/val'
    os.makedirs(val_folder)
    jpg_files = sorted([f for f in os.listdir(folder_path) if f.endswith('HC.png')])

    samples =  random.sample(jpg_files,200)
    for jpg_file in samples:
        img_path = os.path.join(folder_path, jpg_file)
        to_path = os.path.join(val_folder, jpg_file)

        shutil.move(img_path, to_path)
        print(f"Moved: {jpg_file}")
        # img = cv2.imread(img_path)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # if img is None:
        #     print(f"Warning: Image file {jpg_file} could not be loaded.")
        #     continue
        
        annot_file = jpg_file.replace('HC.png', 'HC_Annotation.png')
        annot_path = os.path.join(folder_path, annot_file)
        annot_to_path = os.path.join(val_folder, annot_file)
        shutil.move(annot_path, annot_to_path)
        print(f"Moved: {annot_file}")


def make_filled_mask(folder_path,save_dir): # deprecated
   
    if folder_path == save_dir:
        print('save dir must be different from original folder')
        return None

    else:
        # Create the directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        jpg_files = sorted([f for f in os.listdir(folder_path) if f.endswith('HC.png')])
        count = 0
        for jpg_file in jpg_files:

            src_path = os.path.join(folder_path, jpg_file)
            dst_path = os.path.join(save_dir, jpg_file)
            shutil.copy(src_path,dst_path)

            

            
            annot_file = jpg_file.replace('HC.png', 'HC_Annotation.png')
            annot_path = os.path.join(folder_path, annot_file)
            filled_annot_path = os.path.join(save_dir,annot_file)
            if os.path.exists(annot_path):
                mask = cv2.imread(annot_path)
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                # Create an empty black image to draw the filled circle
                filled_img = np.zeros_like(mask)

                # Fill the contour (circle) with white color
                cv2.drawContours(filled_img, contours, -1, (255), thickness=cv2.FILLED)
                cv2.imwrite(filled_annot_path,filled_img)
                # count +=1

            else:
                print(f"Warning: No annot file found for {jpg_file}")
        

def just_get_ground_truth_sum(filled_annot_path):
    filled_annots = pd.Series()
    jpg_files = sorted([f for f in os.listdir(filled_annot_path) if f.endswith('HC_Annotation.png')])

    for jpg_file in jpg_files:
        img_path = os.path.join(filled_annot_path, jpg_file)
        img = cv2.imread(img_path)
        path_end = jpg_file.replace("HC_Annotation.png","HC.png")
        filled_annots[path_end] = img.sum() /255
    return filled_annots





def load_images_and_masks(folder_path,image_size=(320, 320)): # deprecated
    images = []
    masks = []
    
    jpg_files = sorted([f for f in os.listdir(folder_path) if f.endswith('HC.png')])

    for jpg_file in jpg_files:
        img_path = os.path.join(folder_path, jpg_file)
        img = cv2.imread(img_path)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if img is None:
            print(f"Warning: Image file {jpg_file} could not be loaded.")
            continue
        
        annot_file = jpg_file.replace('HC.png', 'HC_Annotation.png')
        annot_path = os.path.join(folder_path, annot_file)
        
        if os.path.exists(annot_path):
            mask = cv2.imread(annot_path)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Create an empty black image to draw the filled circle
            filled_img = np.zeros_like(mask)

            # Fill the contour (circle) with white color
            cv2.drawContours(filled_img, contours, -1, (1), thickness=cv2.FILLED)

            img_resized = cv2.resize(img, image_size)
            mask_resized = cv2.resize(filled_img, image_size, interpolation=cv2.INTER_NEAREST)
            
            img_resized = img_resized / 255.0
            # mask_resized = mask_resized /255.0
            
            
            images.append(img_resized)
            masks.append(mask_resized[..., np.newaxis]) 
        else:
            print(f"Warning: No annot file found for {jpg_file}")
    
    return np.array(images), np.array(masks)