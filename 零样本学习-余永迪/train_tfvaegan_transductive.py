# author: akshitac8
from __future__ import print_function
import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
# load files
import util
import classifier
import model
from config import opt

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
cudnn.benchmark = True
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
# load data
data = util.DATA_LOADER(opt)
print("training samples: ", data.ntrain)
print("Dataset: ", opt.dataset)

# Init modules: Encoder, Generator, Discriminator
netE = model.Encoder(opt)
netG = model.Generator(opt)
netD = model.Discriminator_D1(opt)
netD2 = model.Discriminator_D2(opt)
# Init models: Feedback module, auxillary module
netF = model.Feedback(opt)
netDec = model.AttDec(opt, opt.attSize)

print(netE)
print(netG)
print(netD)
print(netD2)
print(netF)
print(netDec)

# Init Tensors
input_res_D2 = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att_D2 = torch.FloatTensor(opt.batch_size, opt.attSize)
input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
noise = torch.FloatTensor(opt.batch_size, opt.nz)
one = torch.FloatTensor([1])
mone = one * -1

# Cuda
if opt.cuda:
    netD.cuda()
    netD2.cuda()
    netE.cuda()
    netF.cuda()
    input_res = input_res.cuda()
    noise, input_att = noise.cuda(), input_att.cuda()
    input_res_D2,input_att_D2 = input_res_D2.cuda(), input_att_D2.cuda()
    one = one.cuda()
    mone = mone.cuda()
    netG.cuda()
    netDec.cuda()

def loss_fn(recon_x, x, mean, log_var):
    #vae loss L_bce + L_kl
    BCE = torch.nn.functional.binary_cross_entropy(recon_x+1e-12, x.detach(),size_average=False)
    BCE = BCE.sum()/ x.size(0)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())/ x.size(0)
    return (BCE + KLD)

def WeightedL1(pred, gt):
    #semantic embedding cycle-consistency loss
    wt = (pred-gt).pow(2)
    wt /= wt.sum(1).sqrt().unsqueeze(1).expand(wt.size(0),wt.size(1))
    loss = wt * (pred-gt).abs()
    return loss.sum()/loss.size(0)


def crossLoss(h, true_att):
    h = netF()
    h = torch.reshape(h, opt.attSize)
    true_att = torch.FloatTensor(opt.batch_size, opt.attSize)
    reconstruction_criterion = nn.L1Loss(size_average=False)
    cross_loss = reconstruction_criterion(h, true_att)
    return cross_loss


def feedback_module(gen_out, att, feed_weight=opt.a1, netG=None, netDec=None, netF=None):
    #feedback operation at loop=1
    fake = netG(gen_out, c=att)
    recons = netDec(fake) # function call
    recons_hidden_feat = netDec.getLayersOutDet()
    feedback_out = netF(recons_hidden_feat)
    fake = netG(gen_out, a1=feed_weight, c=att, feedback_layers=feedback_out)
    return fake

def sample():
    #dataloader
    batch_feature, batch_att = data.next_seen_batch(opt.batch_size)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)

def trans_sample():
    #dataloader
    trans_batch_feature, trans_batch_att = data.next_unseen_batch(opt.batch_size)
    input_res_D2.copy_(trans_batch_feature)
    input_att_D2.copy_(trans_batch_att)

def generate_syn_feature(netG, classes, attribute, num, netF=None, netDec=None):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass*num, opt.resSize)
    syn_label = torch.LongTensor(nclass*num) 
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()
    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        syn_noisev = Variable(syn_noise,volatile=True)
        syn_attv = Variable(syn_att,volatile=True)
        output = feedback_module(gen_out=syn_noisev, att=syn_attv, feed_weight=opt.a2, netG=netG, netDec=netDec, netF=netF)
        syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i*num, num).fill_(iclass)
    return syn_feature, syn_label

optimizerD = optim.Adam(netD.parameters(), lr=opt.lr,betas=(opt.beta1, 0.999))
optimizerD2 = optim.Adam(netD2.parameters(), lr=opt.lr,betas=(opt.beta1, 0.999))
optimizerE = optim.Adam(netE.parameters(), lr=opt.lr)
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr,betas=(opt.beta1, 0.999))
optimizerF = optim.Adam(netF.parameters(), lr=opt.feed_lr, betas=(opt.beta1, 0.999))
optimizerDec = optim.Adam(netDec.parameters(), lr=opt.dec_lr, betas=(opt.beta1, 0.999))

def calc_gradient_penalty(netD,real_data, fake_data,input_att=None):
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    if opt.cuda:
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)
    if input_att is None:
        disc_interpolates = netD(interpolates)
    else: 
        disc_interpolates = netD(interpolates, Variable(input_att))
    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    if input_att is None:
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda2
    else:
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty


best_zsl_acc = 0
if opt.gzsl:
    best_gzsl_acc = 0

#Training loop
for epoch in range(0,opt.nepoch):
    #feedback training loop
    for loop in range(0,opt.feedback_loop):
        for i in range(0, data.ntrain, opt.batch_size):
            ############Discriminator D1 ##############
            #unfreeze discrimator
            for p in netD.parameters():
                p.requires_grad = True
                
            #unfreeze decoder
            for p in netDec.parameters():
                p.requires_grad = True
            
            # Train D1 and Decoder
            gp_sum = 0
            for iter_d in range(opt.critic_iter):
                sample()
                netD.zero_grad()          
                input_resv = Variable(input_res)
                input_attv = Variable(input_att)
                
                # Training the auxillary module
                netDec.zero_grad()
                recons = netDec(input_resv)
                h = netF()
                true_att = torch.FloatTensor(opt.batch_size, opt.attSize)
                R_cost = opt.recons_weight*WeightedL1(recons, input_attv) + opt.cross_weight * crossLoss(h, true_att)
                R_cost.backward()
                optimizerDec.step()
                criticD_real = netD(input_resv, input_attv)
                criticD_real = opt.gammaD*criticD_real.mean()
                criticD_real.backward(mone)
                if opt.encoded_noise:        
                    means, log_var = netE(input_resv, input_attv)
                    std = torch.exp(0.5 * log_var)
                    eps = torch.randn([opt.batch_size, opt.latent_size])
                    if opt.cuda: eps = eps.cuda()
                    eps = Variable(eps)
                    z = eps * std + means
                else:
                    noise.normal_(0, 1)
                    z = Variable(noise)

                # feedback loop
                if loop == 1:
                    fake = feedback_module(gen_out=z, att=input_attv, feed_weight=opt.a1, netG=netG, netDec=netDec, netF=netF)
                else:
                    fake = netG(z, c=input_attv)
                criticD_fake = netD(fake.detach(), input_attv)
                criticD_fake = opt.gammaD*criticD_fake.mean()
                criticD_fake.backward(one)
                # gradient penalty
                gradient_penalty = opt.gammaD*calc_gradient_penalty(netD, input_res, fake.data, input_att)
                gp_sum += gradient_penalty.data
                gradient_penalty.backward()         
                Wasserstein_D = criticD_real - criticD_fake
                D_cost = criticD_fake - criticD_real + gradient_penalty  # add Y here and #add vae reconstruction loss
                optimizerD.step()

            #unfreeze discrimator
            for p in netD2.parameters():
                p.requires_grad = True
                
            # Train D2
            for iter_d in range(opt.critic_iter):
                trans_sample()
                netD2.zero_grad()          
                input_resv_D2 = Variable(input_res_D2)
                input_attv_D2 = Variable(input_att_D2)
                criticD2_real = netD2(input_resv_D2)
                criticD2_real = opt.gammaD2*criticD2_real.mean()
                criticD2_real.backward(mone)
                noise.normal_(0, 1)
                noisev = Variable(noise)
                if loop == 1:
                    fake_D2 = feedback_module(gen_out=noisev, att=input_attv_D2, feed_weight=opt.a1, netG=netG, netDec=netDec, netF=netF)
                else:
                    fake_D2 = netG(noisev, c=input_attv_D2)
                criticD2_fake = netD2(fake_D2.detach())
                criticD2_fake = opt.gammaD2*criticD2_fake.mean()
                criticD2_fake.backward(one)
                # gradient penalty
                gradient_penalty_D2 = opt.gammaD2*calc_gradient_penalty(netD2, input_res_D2, fake_D2.data)
                gp_sum += gradient_penalty_D2.data
                gradient_penalty_D2.backward()         
                Wasserstein_D2 = criticD2_real - criticD2_fake
                D2_cost = criticD2_fake - criticD2_real + gradient_penalty_D2 #add Y here and #add vae reconstruction loss
                optimizerD2.step()

            gp_sum /= (opt.gammaD2*opt.lambda1*2*opt.critic_iter)
            if (gp_sum > 1.05).sum() > 0:
                opt.lambda2 *= 1.1
                opt.lambda1 *= 1.1
            elif (gp_sum < 1.001).sum() > 0:
                opt.lambda2 /= 1.1
                opt.lambda1 /= 1.1

            #############netG training ##############
            # Train netG and Decoder
            
            #freeze discrimator
            for p in netD.parameters():
                p.requires_grad = False

            if opt.recons_weight > 0 and opt.freeze_dec:
                #freeze decoder
                for p in netDec.parameters():
                    p.requires_grad = False

            netE.zero_grad()
            netG.zero_grad()
            netF.zero_grad()
            input_resv = Variable(input_res)
            input_attv = Variable(input_att)
            means, log_var = netE(input_resv, input_attv)
            std = torch.exp(0.5 * log_var)
            eps = torch.randn([opt.batch_size, opt.latent_size])
            if opt.cuda: eps = eps.cuda()
            eps = Variable(eps)
            z = eps * std + means
            if loop == 1:
                recon_x = feedback_module(gen_out=z, att=input_attv, feed_weight=opt.a1, netG=netG, netDec=netDec, netF=netF)            
            else:
                recon_x = netG(z, c=input_attv)
            
            #vae reconstruction loss
            vae_loss_seen = loss_fn(recon_x, input_resv, means, log_var) # minimize E
            errG = vae_loss_seen
             
            if opt.encoded_noise:
                criticG_fake = netD(recon_x, input_attv).mean()
                fake = recon_x 
            else:
                noise.normal_(0, 1)
                noisev = Variable(noise)
                if loop == 1:
                    fake = feedback_module(gen_out=noisev, att=input_attv, feed_weight=opt.a1, netG=netG, netDec=netDec, netF=netF)
                else:
                    fake = netG(noisev, c=input_attv)
                criticG_fake = netD(fake, input_attv).mean()

            # Add generator loss
            G_cost = -criticG_fake
            errG += opt.gammaG*G_cost
            
            # Add reconstruction loss
            netDec.zero_grad()
            recons_fake = netDec(fake)
            R_cost = WeightedL1(recons_fake, input_attv)
            errG += opt.recons_weight * R_cost
            
            #freeze discrimator D2
            for p in netD2.parameters():
                p.requires_grad = False

            input_attv_D2 = Variable(input_att_D2)
            noise.normal_(0, 1)
            noisev = Variable(noise)
            if loop == 1:
                fake_D2 = feedback_module(gen_out=noisev, att=input_attv_D2, feed_weight=opt.a1, netG=netG, netDec=netDec, netF=netF)
            else:    
                fake_D2 = netG(noisev, c=input_attv_D2)
            
            criticG_fake_D2 = netD2(fake_D2).mean()
            G_cost_D2 = -criticG_fake_D2
            errG += opt.gammaG_D2*G_cost_D2

            recons_fake_2 = netDec(fake_D2)
            R_cost_D2 = WeightedL1(recons_fake_2, input_attv_D2) #, bce=opt.bce_att, gt_bce=Variable(input_bce_att_D2))
            errG += opt.recons_weight * R_cost_D2
            errG.backward()
            optimizerE.step()
            optimizerG.step()
            if loop == 1:
                optimizerF.step()
            if opt.recons_weight > 0 and not opt.freeze_dec:
                optimizerDec.step()
    print('[%d/%d] Loss_D: %.4f  Loss_D2: %.4f Loss_G: %.4f, Wasserstein_dist:%.4f, vae_loss_seen:%.4f'% (epoch, opt.nepoch, \
    D_cost.data[0],D2_cost.data[0], G_cost.data[0], Wasserstein_D.data[0],vae_loss_seen.data[0]),end=" ")
    netG.eval()
    netDec.eval()
    netF.eval()
    syn_feature, syn_label = generate_syn_feature(netG,data.unseenclasses, data.attribute, opt.syn_num, netF=netF,netDec=netDec)
    # Generalized zero-shot learning
    if opt.gzsl:    
        train_X = torch.cat((data.train_feature, syn_feature), 0)
        train_Y = torch.cat((data.train_label, syn_label), 0)
        nclass = opt.nclass_all
        gzsl_cls = classifier.CLASSIFIER(train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, \
                                    0.5, 25, opt.syn_num, generalized=True, netDec=netDec, dec_size=opt.attSize, dec_hidden_size=4096)
        if best_gzsl_acc < gzsl_cls.H:
            best_acc_seen, best_acc_unseen, best_gzsl_acc = gzsl_cls.acc_seen, gzsl_cls.acc_unseen, gzsl_cls.H
        print('GZSL: seen=%.4f, unseen=%.4f, h=%.4f' % (gzsl_cls.acc_seen, gzsl_cls.acc_unseen, gzsl_cls.H),end=" ")
    # Zero-shot learning
    zsl_cls = classifier.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses), data, data.unseenclasses.size(0), \
                                    opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, generalized=False, netDec=netDec, dec_size=opt.attSize, dec_hidden_size=4096)
    acc = zsl_cls.acc
    if best_zsl_acc < acc:
        best_zsl_acc = acc
    print('ZSL: unseen accuracy=%.4f' % (acc))
    
    # reset models to training mode
    netG.train()
    netDec.train()
    netF.train()

# Best results
print('Dataset', opt.dataset)
print('the best ZSL unseen accuracy is', best_zsl_acc)
if opt.gzsl:
    print('the best GZSL seen accuracy is', best_acc_seen)
    print('the best GZSL unseen accuracy is', best_acc_unseen)
    print('the best GZSL H is', best_gzsl_acc)