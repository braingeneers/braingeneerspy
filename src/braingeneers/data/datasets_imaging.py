import os
import urllib.request, json 
from urllib.error import HTTPError

from skimage import io


camera_ids = [11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 26, 31, 32, 33, 34, 35, 36, 41, 42, 43, 44, 45, 46]

def get_timestamps(uuid):
    with urllib.request.urlopen("https://s3.nautilus.optiputer.net/braingeneers/archive/"+uuid+ "/images/manifest.json") as url:
        data = json.loads(url.read().decode())
        return data['captures']

camera_ids = [11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 26, 31, 32, 33, 34, 35, 36, 41, 42, 43, 44, 45, 46]


def import_json(uuid):
    with urllib.request.urlopen("https://s3.nautilus.optiputer.net/braingeneers/archive/"+uuid+"/images/manifest.json") as url:
        json_file = json.loads(url.read().decode())
        return json_file
    

def save_images(uuid, timestamps = None, cameras=None , focal_lengths=None):
    if os.path.isdir(uuid):
        #the directory is already made
        pass
    else: 
        os.mkdir(uuid)
    
    
    if os.path.isdir(uuid+'/images'):
        #the directory is already made
        pass
    else: 
        os.mkdir(uuid+'/images')
    
    
    
    json_file = import_json(uuid)
    
    if type(timestamps) == int:
        actual_timestamps = [json_file['captures'][timestamps]]
        
    elif type(timestamps) == list:
        actual_timestamps = [json_file['captures'][x] for x in timestamps]
        
    elif timestamps == None:
        actual_timestamps = json_file['captures']
        
    ################################################################
    
    if type(cameras) == int:
        actual_cameras = [cameras]
        
    elif type(cameras) == list:
        actual_cameras = cameras
        
    elif cameras == None:
        actual_cameras = camera_ids
        
    ################################################################
    actual_focal_lengths=[]
    if type(focal_lengths) == int:
        actual_focal_lengths = [focal_lengths]
        
    elif type(focal_lengths) == list:
        actual_focal_lengths = focal_lengths
        
    elif focal_lengths == None:
        actual_focal_lengths = list(range(1, json_file['stack_size']+1))
        
    
    for timestamp in actual_timestamps:
        if os.path.isdir(uuid+'/images/'+str(timestamp)):
            #the directory is already made
            pass
        else: 
            os.mkdir(uuid+'/images/'+str(timestamp))
            
        for camera in actual_cameras:
            if os.path.isdir(uuid+'/images/'+str(timestamp)+"/cameraC" + str(camera)):
                #the directory is already made
                pass
            else: 
                os.mkdir(uuid+'/images/'+str(timestamp)+"/cameraC" + str(camera))
                
            for focal_length in actual_focal_lengths:
                
                full_path = "https://s3.nautilus.optiputer.net/braingeneers/archive/"+uuid+'/images/'+str(timestamp)+"/cameraC"\
                + str(camera) +"/"+str(focal_length)+".jpg"
                
                io.imsave(uuid+'/images/'+str(timestamp)+"/cameraC"+ str(camera) +"/"+str(focal_length)+".jpg", io.imread(full_path))
                
                print('Downloading image to: '+ str(uuid)+'/images/'+str(timestamp)+"/cameraC"\
                + str(camera) +"/"+str(focal_length)+".jpg")
                
                try:
                    io.imshow(io.imread(full_path))
                    
                except HTTPError as err:
                   if err.code == 403:
                       print("URL address is wrong.")

    
    return None

        
