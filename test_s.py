from options import TestOptions
import torch
from models import GlyphGenerator, TextureGenerator
from utils import load_image, to_data, to_var, visualize, save_image, gaussian
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def concat_h(list_1d):
      # function calling for every 
    # list of images
    
    images = []
    max_height = 0 # find the max width of all the images
    total_width = 0 # the total height of the images (vertical stacking)

    for im in list_1d:
        # open all images and find their sizes
        images.append(im)
        
        if images[-1].shape[0] > max_height:
            max_height = images[-1].shape[0]
        total_width += images[-1].shape[1]
    
    # create a new array with a size large enough to contain all the images
    final_image = np.zeros((max_height,total_width,3),dtype=np.uint8)
    print('finalh',final_image.shape)
    current_x = 0 # keep track of where your current image was last placed in the y coordinate
    for image in images:
        print(image.shape)
        # add an image to the final array and increment the y coordinate
        final_image[:image.shape[0],current_x:image.shape[1]+current_x,:] = image
        current_x += image.shape[1]
      
    # return final image
    return final_image
def concat_v(list_1d):
      # function calling for every 
    # list of images
    
    images = []
    max_width = 0 # find the max width of all the images
    total_height = 0 # the total height of the images (vertical stacking)

    for im in list_1d:
        # open all images and find their sizes
        images.append(im)
        
        if images[-1].shape[1] > max_width:
            max_width = images[-1].shape[1]
        total_height += images[-1].shape[0]
    
    # create a new array with a size large enough to contain all the images
    final_image = np.zeros((total_height,max_width,3),dtype=np.uint8)
    print('finalv',final_image.shape)
    current_y = 0 # keep track of where your current image was last placed in the y coordinate
    for image in images:
        print(image.shape)
        # add an image to the final array and increment the y coordinate
        final_image[current_y:image.shape[0]+current_y,:image.shape[1],:] = image
        current_y += image.shape[0]
      
    # return final image
    return final_image

def main():
    # parse options
    
    # for ep in range(1,101):
    parser = TestOptions()
    opts = parser.parse()
    ep = opts.model_n
    # data loader
    print('--- load data ---')
    text = load_image(opts.text_name, opts.text_type)
    if opts.gpu:
        text = to_var(text)
    D = 'D:\work\Masterdegree\shapmatching\predict\models'
    C = './predict/models'
    # model
    print('--- load model ---')
    netGlyph = GlyphGenerator(n_layers=6, ngf=32)
    netGlyph.load_state_dict(torch.load(os.path.join(os.path.join(C,opts.structure_model),(str(ep) +'-GS.ckpt'))))
    if opts.gpu:
        netGlyph.cuda()
    netGlyph.eval()
    img_str = []
    
    print('--- testing ---')
    text[:,0:1] = gaussian(text[:,0:1], stddev=0.2)
    sub = opts.sub_image
    ori_ht = (text.size())[2]
    ori_wd = (text.size())[3]
    m = 0
    n = 0
    a = 0
    b = 0
    for y in range(0,ori_ht,sub):
        a = 0
        for x in range(0,ori_wd,sub):
            a += 1
        b += 1
    print(b,a)
    img_list = np.empty((b,a),dtype = object)
    print('create',img_list.shape)
    for y in range(0,ori_ht,sub):
        n = 0
        for x in range(0,ori_wd,sub):
            text_split = text[:,:,y:y+sub,x:x+sub].clone()
            a = (text_split.cpu().detach().numpy())
            a = np.squeeze(a)
            a = np.concatenate((np.expand_dims(a.squeeze()[0,:,:],axis=-1),np.expand_dims(a.squeeze()[1,:,:],axis=-1),np.expand_dims(a.squeeze()[2,:,:],axis=-1)),axis=-1)
            # plt.imshow(a)
            # plt.show()
            
            img_str= netGlyph(text_split, 2.0-1.0) 
            img_str = to_data(img_str)
            
            # save_image(img_str[0], result_filename)
            tmp = ((img_str[0].numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)
            # plt.imshow(tmp)
            # plt.show()
            showimg = cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR)
            # print(showimg.shape)
            # img_list.append(showimg)
            img_list[m][n] = showimg
            
            n += 1
        m += 1
    
    r = m
    c = n
    print(round(ori_ht/sub))
    print(round(ori_wd/sub))
    print(r,c)
    # print(img_list[0].shape)
    # if r == 1:
    #     im = concat_h(img_list)
    #     final = im
    # elif c == 1:
    #     im = concat_v(img_list)
    #     final = im
    # else:
        
    img_list = np.asarray(img_list)
    arr = img_list.reshape(r,c)
    # arr = np.array(img_list)
    print(arr.shape)
    h_stack = []
    for l in range(arr.shape[1]):
        im = concat_v(arr[:,l])
        # plt.imshow(im)
        # plt.show()
        h_stack.append(im)
        print(im.shape)
    final = cv2.hconcat(h_stack)

    result_filename = os.path.join(opts.result_dir, (str(opts.pic)+ '_' + str(ep) + '_' + opts.name+'.png'))
    cv2.imwrite(result_filename,final)
    
    # cv2.destroyAllWindows()
    print("text",text[:,0:1].size())
    
        
    print('--- save ---')
    # directory
    if not os.path.exists(opts.result_dir):
        os.mkdir(opts.result_dir)         
    # for i in range(len(result)):     
    #     result_filename = os.path.join(opts.result_dir, (str(i) + '_' + str(ep) + '_' + opts.name+'.png'))
    # save_image(result[i][0], result_filename)

if __name__ == '__main__':
    main()
