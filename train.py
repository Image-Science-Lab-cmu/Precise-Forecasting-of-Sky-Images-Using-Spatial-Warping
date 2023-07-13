import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from LiteFlowNet import *
from losses import *
from skynet_Unet_model import *
from dataLoader import *
from torch.utils.data import DataLoader
import argparse
import numpy as np

import cupy

parser = argparse.ArgumentParser()

parser.add_argument('--input_channels',
                    default=12,
                    type=np.int,
                    help='(default value: %(default)s) Number of channels for input images. 3*NumOfImages')
parser.add_argument('--output_channels',
                    default=3,
                    type=np.int,
                    help='(default value: %(default)s) Number of channels for output images.')
parser.add_argument('--lam_int',
                    default=5.0,
                    type=np.float32,
                    help='(default value: %(default)s) Hyperparameter for intensity loss.')
parser.add_argument('--lam_gd',
                    default=0.00111,
                    type=np.float32,
                    help='(default value: %(default)s) Hyperparameter for gradient loss.')
parser.add_argument('--lam_op',
                    default=0.010,
                    type=np.float32,
                    help='(default value: %(default)s) Hyperparameter for optical flow loss.')
parser.add_argument('--EPOCHS',
                    default=40,
                    type=np.int,
                    help='(default value: %(default)s) Number of epochs o train model for.')
parser.add_argument('--BATCH_SIZE',
                    default=8,
                    type=np.int,
                    help='(default value: %(default)s) Training batch size.')
parser.add_argument('--LR',
                    default=0.0002,
                    type=np.float32,
                    help='(default value: %(default)s) learning rate.')

args = parser.parse_args()


#Model Paths
lite_flow_model_path='./network-sintel.pytorch'

INPUTS_PATH = "./SkyNet_Data/xTrain_skip.h5"
TARGET_PATH = "./SkyNet_Data/yTrain_skip.h5"


# Models
devCount = torch.cuda.device_count()
dev = torch.cuda.current_device()

if devCount > 1:
    dev = "cuda:" + str(devCount - 1)

device = torch.device(dev if torch.cuda.is_available() else "cpu")


# SkyNet UNet
generator = SkyNet_UNet(args.input_channels, args.output_channels)

generator = torch.nn.DataParallel(generator, device_ids=[devCount - 1, 0]) 
generator = generator.to(device)

# Optical Flow Network
flow_network = Network()
flow_network.load_state_dict(torch.load(lite_flow_model_path))
flow_network.cuda().eval()


trainLoader = DataLoader(DatasetFromFolder(INPUTS_PATH, TARGET_PATH), args.BATCH_SIZE, shuffle=True)

# Trains model (on training data) and returns the training loss
def run_train(model, x, y, gd_loss, op_loss, int_loss, optimizer): # Add skip as a parameter here
    target = y
    model = model.train()

    G_output = model(x)
  
    # For Optical Flow
    inputs = x
    input_last = inputs[:, 9:,:,:].clone().cuda() #I_t

    
    pred_flow_esti_tensor = torch.cat([input_last, G_output],1) #(Predicted)
    gt_flow_esti_tensor = torch.cat([input_last, target],1) #(Ground Truth)
    

    flow_gt = batch_estimate(gt_flow_esti_tensor, flow_network)
    flow_pred = batch_estimate(pred_flow_esti_tensor, flow_network)

    
    g_op_loss = op_loss(flow_pred, flow_gt)
    g_int_loss = int_loss(G_output, target)
    g_gd_loss = gd_loss(G_output, target)

    g_loss = args.lam_gd*g_gd_loss + args.lam_op*g_op_loss + args.lam_int*g_int_loss

    optimizer.zero_grad()

    g_loss.backward()
    optimizer.step()
    
    return g_loss.item()




# Sochastic Gradient Descent Weight Updater
optimizer = optim.Adam(generator.parameters(), lr = args.LR)


# Training Loss
train_loss = []

# Validation Loss 
valid_loss = []

# Losses
gd_loss = Gradient_Loss(1, 3).to(device)
op_loss = Flow_Loss().to(device)
int_loss = Intensity_Loss(1).to(device)

# Training Part ...
num_images = 0
for epoch in tqdm(range(args.EPOCHS), position = 0, leave = True):
    print('Starting Epoch...', epoch + 1)
    
    trainLossCount = 0
    num_images = 0
    for i, data in enumerate(trainLoader):
        # Training
        inputs = Variable(data[0]).to(device) # The input data
        target = Variable(data[1]).float().to(device)
        
        num_images += inputs.size(0)
        
        # Trains model
        trainingLoss = run_train(generator, inputs, target, gd_loss, op_loss, int_loss, optimizer) # Add skip as a parameter here
        trainLossCount = trainLossCount + trainingLoss
        
    
    epoch_loss = trainLossCount/num_images
    train_loss.append(epoch_loss)

    print('Training Loss...')
    print("===> Epoch[{}]({}/{}): Loss: {:.8f}".format(epoch + 1, i + 1, len(trainLoader), epoch_loss))

    if epoch % 10 == 0:
        PATH =  './Iteration' + str(epoch) + '.pt'
        torch.save(generator, PATH)
        print('Saved model iteration' +  str(epoch) + 'to -> ' + PATH)


print('Training Complete...')

PATH =  './finalModel.pt'
torch.save(generator, PATH)
print('Saved Finnal Model to -> ' + PATH)