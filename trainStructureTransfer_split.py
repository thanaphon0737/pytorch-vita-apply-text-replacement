from __future__ import print_function
import torch
from models import SketchModule, ShapeMatchingGAN, GlyphGenerator
from utils import load_image, to_data, to_var, visualize, save_image, gaussian, weights_init
from utils import load_train_batchfnames, prepare_text_batch, load_style_image_pair
from utils2 import cropping_training_batches
import random
from options import TrainShapeMatchingOptions
import numpy as np
import os
import time
import pickle
import pandas as pd
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
    # print('finalh',final_image.shape)
    current_x = 0 # keep track of where your current image was last placed in the y coordinate
    for image in images:
        # print(image.shape)
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
    # print('finalv',final_image.shape)
    current_y = 0 # keep track of where your current image was last placed in the y coordinate
    for image in images:
        # print(image.shape)
        # add an image to the final array and increment the y coordinate
        final_image[current_y:image.shape[0]+current_y,:image.shape[1],:] = image
        current_y += image.shape[0]
      
    # return final image
    return final_image
def main():
    # parse options
    parser = TrainShapeMatchingOptions()
    opts = parser.parse()
    start_time = time.time()
    # create model
    print('--- create model ---')
    netShapeM = ShapeMatchingGAN(opts.GS_nlayers, opts.DS_nlayers, opts.GS_nf, opts.DS_nf,
                     opts.GT_nlayers, opts.DT_nlayers, opts.GT_nf, opts.DT_nf, opts.gpu,opts.l1,opts.sadv)


    if opts.gpu:
        netShapeM.cuda()

    netShapeM.init_networks(weights_init)
    netShapeM.train()

    print('--- training ---')
    # load image pair
    scales = 0
    Xl, X, _, Noise = load_style_image_pair(opts.style_name,opts.std_name, scales,None, opts.gpu)
    print('XL' , len(Xl), Xl[0].shape)
    print('X', X.shape)
    print('Noise', Noise.shape)
    Xl = [to_var(a) for a in Xl] if opts.gpu else Xl
    X = to_var(X) if opts.gpu else X
    Noise = to_var(Noise) if opts.gpu else Noise
    epoch_csv = []
    LDadv = []
    LGadv = []
    Lrec = []
    Lgly = []
    LDadvt = []
    LGadvt = []
    Lrect = []
    os.makedirs(os.path.join('./predict',opts.save_name), exist_ok=True)
    os.makedirs(os.path.join('./predict/models',opts.save_name), exist_ok=True)
    
    for epoch in range(opts.step1_epochs):
        for i in range(opts.batchsize):
            idx = 0
            xl, x = cropping_training_batches(Xl[idx], X, Noise, opts.batchsize, 
                                      opts.Sanglejitter, opts.subimg_size, opts.subimg_size,i)
            losses = netShapeM.structure_one_pass(x, xl, 0)
            print('Step1, Epoch [%02d/%02d][%03d/%03d]' %(epoch+1, opts.step1_epochs, i+1, 
                                                          opts.Straining_num//opts.batchsize), end=': ')
            print('LDadv: %+.3f, LGadv: %+.3f, Lrec: %+.3f, Lgly: %+.3f'%(losses[0], losses[1], losses[2], losses[3]))
            LDadvt.append(losses[0].detach().cpu().numpy())
            LGadvt.append(losses[1].detach().cpu().numpy())
            Lrect.append(losses[2].detach().cpu().numpy())
            # if i == (opts.Straining_num//opts.batchsize) - 1:
            #     print('can save')
            #     # losses_save.append([losses[0].detach().cpu().numpy(), losses[1].detach().cpu().numpy(), losses[2].detach().cpu().numpy(), losses[3]])
            #     epoch_csv.append(epoch+1)
            #     LDadv.append(losses[0].detach().cpu().numpy())
            #     LGadv.append(losses[1].detach().cpu().numpy())
            #     Lrec.append(losses[2].detach().cpu().numpy())
            #     Lgly.append(losses[3])
        epoch_csv.append(epoch+1)
        LDadv.append(np.mean(np.asarray(LDadvt)))    
        LGadv.append(np.mean(np.asarray(LGadvt)))  
        Lrec.append(np.mean(np.asarray(Lrect)))  
        Lgly.append(losses[3])

        # if epoch%num_save_epoch ==0:
        netShapeM.save_structure_model(os.path.join('./predict/models',opts.save_name),  str(epoch+1))  
        netGlyph = GlyphGenerator(n_layers=6, ngf=32)
        netGlyph.load_state_dict(torch.load('./predict/models/'+ opts.save_name +'/' +str(epoch+1)+ '-GS.ckpt'))
        
        if opts.gpu:
            netGlyph.cuda()
        netGlyph.eval()
        print('--- testing ---')
        text = load_image(opts.text_name, 0)
        if opts.gpu:
            text = to_var(text) 
        text[:,0:1] = gaussian(text[:,0:1], stddev=0.2)
        sub = opts.subimg_size
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
        if True:
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

        result_filename = os.path.join('./predict/' + opts.save_name, (str(epoch+1) + '_' + opts.save_name +'.png'))
        cv2.imwrite(result_filename,final)
    save_csv = {
        'epoch':epoch_csv,
        'LDadv':LDadv,
        'LGadv':LGadv,
        'Lrec':Lrec,
        'Lgly':Lgly
    }
    df = pd.DataFrame(save_csv)
    df.to_csv(r'./losses/' + opts.save_name + '.csv',index=False)
    # with open('./losses/' + opts.save_name + '.pkl', 'wb') as file:
                
    #             # A new file will be created
    #             pickle.dump(losses_save, file)


    print('--- save ---')
    # directory
    netShapeM.save_structure_model(opts.save_path, opts.save_name)  

    # record time to path ./time
    sec = time.time() - start_time
    minit = sec/60
    hour = minit/60
    print("--- {:.3f} seconds  ---".format(sec))
    print("--- {:.3f} minits  ---".format(minit))
    print("--- {:.3f} hours  ---".format(hour))
    time_path = opts.save_name + '_' + str(epoch)
    time_save_path = './rec_time/'+time_path + '_' + str(opts.Straining_num) +'-StructureTrans' + '.txt'
    with open(time_save_path, 'w') as the_file:
        the_file.write("--- path {}  ---\n".format(time_save_path))
        the_file.write("--- path_save {}  ---\n".format(opts.save_path))
        the_file.write("--- {:.3f} seconds  ---\n".format(sec))
        the_file.write("--- {:.3f} minits  ---\n".format(minit))
        the_file.write("--- {:.3f} hours  ---\n".format(hour))

if __name__ == '__main__':
    main()
