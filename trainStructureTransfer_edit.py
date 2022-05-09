from __future__ import print_function
import torch
from models import SketchModule, ShapeMatchingGAN, GlyphGenerator
from utils import load_image, to_data, to_var, visualize, save_image, gaussian, weights_init
from utils import load_train_batchfnames, prepare_text_batch, load_style_image_pair, cropping_training_batches
import random
from options import TrainShapeMatchingOptions
import os
import time
import pickle
import pandas as pd
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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
    os.makedirs(os.path.join('./predict',opts.save_name), exist_ok=True)
    os.makedirs(os.path.join('./predict/models',opts.save_name), exist_ok=True)
    num_save_epoch = int((opts.step1_epochs-1)/40)
    print(num_save_epoch)
    for epoch in range(opts.step1_epochs):
        for i in range(opts.Straining_num//opts.batchsize):
            idx = 0
            xl, x = cropping_training_batches(Xl[idx], X, Noise, opts.batchsize, 
                                      opts.Sanglejitter, opts.subimg_size, opts.subimg_size)
            losses = netShapeM.structure_one_pass(x, xl, 0)
            print('Step1, Epoch [%02d/%02d][%03d/%03d]' %(epoch+1, opts.step1_epochs, i+1, 
                                                          opts.Straining_num//opts.batchsize), end=': ')
            print('LDadv: %+.3f, LGadv: %+.3f, Lrec: %+.3f, Lgly: %+.3f'%(losses[0], losses[1], losses[2], losses[3]))
            if i == (opts.Straining_num//opts.batchsize) - 1:
                print('can save')
                # losses_save.append([losses[0].detach().cpu().numpy(), losses[1].detach().cpu().numpy(), losses[2].detach().cpu().numpy(), losses[3]])
                epoch_csv.append(epoch+1)
                LDadv.append(losses[0].detach().cpu().numpy())
                LGadv.append(losses[1].detach().cpu().numpy())
                Lrec.append(losses[2].detach().cpu().numpy())
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
        print("text",text.size())
        img_str = netGlyph(text, 0*2.0-1.0) 
        result = [img_str]
        if opts.gpu:
            for i in range(len(result)):              
                result[i] = to_data(result[i])
        torch.cuda.empty_cache()
        print('--- save ---')
        # directory
            
        for i in range(len(result)):     
            result_filename = os.path.join('./predict/' + opts.save_name, (str(epoch+1) + '_' + opts.save_name +'.png'))
            save_image(result[i][0], result_filename)
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
