from templates import *

if __name__ == '__main__':
    gpus = [0]

    conf = Example_Diffusion_Model()
    train(conf, gpus=gpus)
    

