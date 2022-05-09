from options import TestOptions
import torch
from models import GlyphGenerator, TextureGenerator
from utils import load_image, to_data, to_var, visualize, save_image, gaussian
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    # parse options
    for ep in range(1,101):
        parser = TestOptions()
        opts = parser.parse()

        # data loader
        print('--- load data ---')
        text = load_image(opts.text_name, opts.text_type)
        if opts.gpu:
            text = to_var(text)
        
        # model
        print('--- load model ---')
        netGlyph = GlyphGenerator(n_layers=6, ngf=32)
        netGlyph.load_state_dict(torch.load(os.path.join(opts.structure_model,(str(ep) +'-GS.ckpt'))))
        if opts.gpu:
            netGlyph.cuda()
        netGlyph.eval()
        
        print('--- testing ---')
        text[:,0:1] = gaussian(text[:,0:1], stddev=0.2)
        print("text",text.size())
        img_str = netGlyph(text, 2.0-1.0) 
        result = [img_str]
        if opts.gpu:
            for i in range(len(result)):              
                result[i] = to_data(result[i])
            
        print('--- save ---')
        # directory
        if not os.path.exists(opts.result_dir):
            os.mkdir(opts.result_dir)         
        for i in range(len(result)):     
            result_filename = os.path.join(opts.result_dir, (str(ep) + '_' + opts.name+'.png'))
        save_image(result[i][0], result_filename)

if __name__ == '__main__':
    main()
