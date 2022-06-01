"""
Script to run regression on the contiguous US precipitation data.
"""
# IMPORTS
import os
import sys
import getopt
import math
import numpy as np
import dill
import torch
import gpytorch
from matplotlib import pyplot as plt
import pyproj
import netCDF4
from mpl_toolkits import basemap
from mpl_toolkits.basemap import cm
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# for plotting
map_of_months = {'20210101' : 'Dec 2020',
                 '20210201' : 'Jan 2021',
                 '20210301' : 'Feb 2021',
                 '20210401' : 'Mar 2021',
                 '20210501' : 'April 2021'
                 }

def plot_from_tensor(output_tensor, template_masked_array, lons, lats, datestr, auto_levels=False):
    """Given 1D tensor fo data output_tensor, fill into valid part of a masked array like template_masked_array and
    plot against lons, lats on fig, returns fig."""
    data_array = np.zeros(template_masked_array.shape)
    data_array[~template_masked_array.mask] = output_tensor.detach().numpy()
    data_array = np.ma.masked_array(data_array, template_masked_array.mask)

    m = basemap.Basemap(projection='stere', lon_0=-105, lat_0=90., lat_ts=60,
            llcrnrlat=lats[0,0], urcrnrlat=lats[-1,-1],
            llcrnrlon=lons[0,0], urcrnrlon=lons[-1,-1],
            rsphere=6371200., resolution='l', area_thresh=10000)
    m.drawcoastlines()
    m.drawstates()
    m.drawcountries()
    parallels = np.arange(0., 90., 10.)
    m.drawparallels(parallels, labels=[1,0,0,0], fontsize=10)
    meridians = np.arange(100., 360., 10.)
    m.drawmeridians(meridians, labels=[0,0,0,1], fontsize=10)

    _, _, x, y, = m.makegrid(data_array.shape[1], 
                                            data_array.shape[0], 
                                            returnxy=True)
    if auto_levels:
        cs = m.contourf(x, y[::-1], data_array, levels=24, cmap=cm.s3pcpn, alpha=0.8)
    else:
        clevs = [0,1,2.5,5,7.5,10,15,20,30,40,50,70,100,150,200,250,300,400,500,600,750]
        cs = m.contourf(x, y[::-1], data_array, clevs, cmap=cm.s3pcpn, alpha=0.8)

    cbar = m.colorbar(cs,location='bottom', pad='5%')
    cbar.set_label('mm')


def parse_args(argv):
    """Get command line arguments, set defaults and return a dict of settings."""

    args_dict = {
        ## File paths and logging
            'data'          : '/data/',             # relative path to data dir from root
            'root'          : '.',                 # path to root directory from which paths are relative
            'device'        : 'cpu',               # cpu or cuda
            'logdir'        : 'experiments/logs/', # relative path to logs from root
            'log_interval'  : 1,                   # how often to log train data
            'test_interval' : 1,                   # how often to log test metrics
            'plot_interval' : 10,                  # how often to generate plots
            'name'          : None,                # name for experiment
            'test_type'      : 'random',           # random or 'censored'
        
        ## Training options
            'model'         : 'DiagonalML',        # 'DiagonalML', 'FullML', 'DiagonalGibbs'
            'datestr'       : '20210101',          # '20210x01' for x in 1, 2, 3, 4
            'inference'     : 'exact',             # 'exact' or 'sparse'
            'train_percent' : '80',                # percentage of data to use for training
            'lr'            : '1e-1',              # learning rate          
            'max_iters'     : 1000,
            'threshold'     : 1e-6,                # improvement after which to stop
            'seed'          : 12,
            'M'             : 1000,                  # Number of inducing points (sparse regression only)
            'prior_scale'   : 1,                    # initial value for the prior outputscale (same for both dims)
            'prior_ell'     : 1.5,                  # Initial value for the prior's lengthscale (same for both dims)
            'prior_mean'    : 0.3,                  # Initial value for the prior's mean (same for both dims)
            'noise'         : 0,                     # 0 for optimised noise, else fixed through training
            'scale'         : 0                     # 0 for optimised output scale, else fixed through training
    }

    try:
        opts, _ = getopt.getopt(argv, '', [name + '=' for name in args_dict.keys()])
    except getopt.GetoptError:
        helpstr = 'Check options. Permitted long options are '
        print(helpstr, args_dict.keys())

    for opt, arg in opts:
        opt_name = opt[2:] # remove initial --
        args_dict[opt_name] = arg
    
    return args_dict

if __name__ == '__main__':

    # ----------------------------------------------------------------------------------- SETTINGS, RANDOM SEED, SETUP
    # ------------------------------------------------------------------------------------------------------------------
    torch.set_default_dtype(torch.float64)
     
    args = parse_args(sys.argv[1:])
    device = args['device']
    torch.manual_seed(int(args['seed']))
    rng = np.random.default_rng(int(args['seed']))

    if args['model'] == 'DiagonalGibbs':
        max_cg_iterations = 4000 # allow more CG iterations in gpytorch's approximate linear solves
    else:
        max_cg_iterations = 1000 # this is the usual default

    os.chdir(args['root'])
    sys.path.append(os.getcwd())
    # local imports...
    from src import models, utils, kernels

    
    # set up the logging directory
    if args['name'] is None:
        args['name'] = ('precipitation_' + args['datestr'] + '_' + args['test_type'] + '_' + args['model'] + '_' 
                        + args['inference'] 
                        )
        if args['model'] == 'DiagonalGibbs':
            args['name'] = (args['name'] + '_scale_' + args['prior_scale'] + '_ell_' + args['prior_ell'] 
                                + '_mean_' + args['prior_mean'])
        args['name'] = args['name'] + '_' + args['device'] + '_' + str(args['seed'])
    done = False
    suffix = 0
    while not done:
        try:
            logdirname = args['root'] + args['logdir'] + args['name'] + '_' + str(suffix)
            os.mkdir(logdirname)
            done = True
        except FileExistsError:
            suffix += 1
    
    # --------------------------------------------------------------------------- DATA LOADING AND NORMALISATION
    # ------------------------------------------------------------------------------------------------------------------

    # load the data, normalise and do train/test split
    nc = netCDF4.Dataset(args['root'] + args['data'] + 'nws_precip_mtd_' + args['datestr'] + '_conus.nc')
    y_grid = 25.4 * nc.variables['normal'][:][::10,::10]

    x1, x2 = nc.variables['x'][:][::10], nc.variables['y'][::-1][::10]
    x1_grid, x2_grid = np.meshgrid(x1, x2)
    projection = pyproj.Proj(projparams='+proj=stere +lat_0=90 '
                                    '+lat_ts=60 +lon_0=-105 +x_0=0 +y_0=0 +a=6371200 +b=6371200 +units=m +no_defs'
                            )
    lons, lats = projection(x1_grid, x2_grid, inverse=True)

    x = torch.tensor(np.vstack((lons[~y_grid.mask].flatten(), lats[~y_grid.mask].flatten())).T).to(device)
    y = torch.tensor(y_grid[~y_grid.mask].flatten()).to(device)
    x = x.double()
    y = y.double()
    with torch.no_grad():
        stdx, meanx = torch.std_mean(x, dim=-2)
        x = (x -  meanx) / stdx
        stdy, meany = torch.std_mean(y)
        y = (y - meany) / stdy

    if args['test_type'] == 'censored':
        # old censor: switch to new on 14.05.2022. Keep this code around to reproduce plots from old runs.
        #censor = ~((torch.tensor(lats) > torch.tensor(40)) * (torch.tensor(lats) < torch.tensor(48))
        #  *(torch.tensor(lons) > -95) * (torch.tensor(lons) < -88)
        #  + (torch.tensor(lats) < torch.tensor(36))
        #  *(torch.tensor(lons) > -105) * (torch.tensor(lons) < -98)
        #  )
        censor = ~((torch.tensor(lats) > torch.tensor(40)) * (torch.tensor(lats) < torch.tensor(48))
          *(torch.tensor(lons) > -95) * (torch.tensor(lons) < -88)

          )
        train_mask = (censor * ~torch.tensor(y_grid.mask)).numpy()
        test_mask = (~censor * ~torch.tensor(y_grid.mask)).numpy()
        normalised_array_1 = np.zeros(y_grid.shape)
        normalised_array_2 = np.zeros(y_grid.shape)
        normalised_array_1[~y_grid.mask] = x[..., 0].detach().numpy()
        normalised_array_2[~y_grid.mask] = x[..., 1].detach().numpy()
        x_train = torch.tensor(np.vstack((normalised_array_1[train_mask].flatten(), 
                                            normalised_array_2[train_mask].flatten())).T).to(device)
        x_test = torch.tensor(np.vstack((normalised_array_1[test_mask].flatten(), 
                                            normalised_array_2[test_mask].flatten())).T).to(device)
        normalised_array_1[~y_grid.mask] = y.detach().numpy()
        y_train = torch.tensor(normalised_array_1[train_mask].flatten()).to(device)                                 
        y_test = torch.tensor(normalised_array_1[test_mask].flatten()).to(device)                                 

    else:
        num_train = math.ceil(float(args['train_percent'])/100 * y.shape[0])
        idx = np.arange(0, y.shape[0], 1)
        rng.shuffle(idx)
        train_idx = idx[:num_train]
        test_idx = idx[num_train:]
        x_train = x[..., train_idx, :].detach()
        y_train = y[..., train_idx].detach()
        x_test = x[..., test_idx, :].detach()
        y_test = y[..., test_idx].detach()

    # ---------------------------------------------------------------------------------------- INITIALISE MODEL
    # ----------------------------------------------------------------------------------------------------------------
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

    if args['inference'] == 'sparse':
        # initialise inducing inputs
        method = utils.init_methods.Kmeans()
        z = torch.tensor(method(x_train.cpu().numpy(), int(args['M']), kernel=None)[0])

    if args['model'] == 'DiagonalML' and args['inference'] == 'exact':
        model = models.stationary.DiagonalExactGP(x_train, y_train, likelihood).to(device)
    elif args['model'] == 'DiagonalGibbs':
        prior = kernels.gibbs.LogNormalPriorProcess(input_dim=2).to(device)
        #### change the prior settings here if desired
        prior.covar_module.outputscale = float(args['prior_scale']) * torch.ones_like(prior.covar_module.outputscale)
        prior.covar_module.base_kernel.lengthscale = float(args['prior_ell']) * torch.ones_like(
                                                                prior.covar_module.base_kernel.lengthscale)
        prior.mean_module.constant = torch.nn.Parameter(
            math.log(float(args['prior_mean'])) * torch.ones_like(prior.mean_module.constant)
                                )
        if args['inference'] == 'exact':
            model = models.nonstationary.DiagonalExactGP(x_train, y_train, likelihood, prior, num_dim=2).to(device)
        elif args['inference'] == 'sparse':
            model = models.nonstationary.DiagonalSparseGP(x_train, y_train, likelihood, prior, z, num_dim=2).to(device)
        else:
            raise NotImplementedError('Not yet implemented {} inference for {}'.format(args['inference'], 
                                                                                                args['model']))
    else:
        raise NotImplementedError('Not yet implemented {} inference for {}'.format(args['inference'], args['model']))

    # Dump initialisation details to a logfile
    stdout = sys.stdout
    with open(logdirname + '/log.txt', 'w') as logfile:
        sys.stdout = logfile
        print('Command run:')
        print(sys.argv)
        print('args used:')
        for key, val in args.items():
            print(key + '\t', val)
        sys.stdout = stdout
    
    # Save the initial model - just in case we e.g. lose the run script or sth like this
    torch.save(model, logdirname + '/init_model.pt',  pickle_module=dill)

    model.train()
    likelihood.train()

    lml = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    optim = torch.optim.Adam(model.parameters(), lr=float(args['lr']))

    writer = SummaryWriter(log_dir = logdirname + '/tensorboard')

    # in case of issues with requires_grad status:
    for group in optim.param_groups:
        for p in group['params']:
            p.requires_grad = True
        
    # for the nonstationary case, make sure the prior's kernel is not being optimised
    if args['model'] == 'DiagonalGibbs':
        for p in prior.parameters():
            p.requires_grad = False
    
    # noise hyper
    if float(args['noise']) > 0:
        model.likelihood.noise = float(args['noise'])
        for p in model.likelihood.noise_covar.parameters():
            p.requires_grad = False
    # outputscale hyper
    if float(args['scale']) > 0:
        model.covar_module.outputscale = float(args['scale'])
        model.covar_module._parameters['raw_outputscale'].requires_grad = False

    # --------------------------------------------------------------------------------------------- TRAINING
    # ----------------------------------------------------------------------------------------------------------------

    change = 2*float(args['threshold'])
    best = math.inf
    best_rmse = math.inf
    best_nlpd = math.inf
    loss = best

    for i in tqdm(range(int(args['max_iters']))):
        optim.zero_grad()
        old_loss = loss
        with gpytorch.settings.max_cg_iterations(max_cg_iterations):
            output = model(x_train)
            loss = - lml(output, y_train)
        change = torch.abs(old_loss-loss).item()
        loss.backward()

        if i % int(args['log_interval']) == 0:
            writer.add_scalar('Objective', loss, i)
            writer.add_scalar('hypers/noise', likelihood.noise.squeeze(), i)
            writer.add_scalar('hypers/outputscale', model.covar_module.outputscale.squeeze(), i)
            if not args['model'] == 'DiagonalGibbs':
                for d in range(x_train.shape[-1]):
                    writer.add_scalar('hypers/lengthscale_{}'.format(d+1), 
                                                model.covar_module.base_kernel.lengthscale.squeeze()[d], i
                                                )

        if i % int(args['test_interval']) == 0:
            model.eval()
            likelihood.eval()
            if args['model'] == 'DiagonalGibbs':
                pred_y = likelihood(model.predict(x_test))
            else:
                pred_y = likelihood(model(x_test))
            rmse = utils.metrics.sqrt_mean_squared_error(y_test, pred_y.mean)
            nlpd = utils.metrics.negative_log_predictive_density(y_test, pred_y.mean, 
                                    torch.diagonal(pred_y.covariance_matrix, dim1=-2, dim2=-1))
            writer.add_scalar('test/rmse', rmse, i)
            writer.add_scalar('test/nlpd', nlpd, i)
            writer.add_scalar('test_rescaled/rmse', rmse*stdy, i)
            writer.add_scalar('test_rescaled/nlpd', nlpd + torch.log(stdy.squeeze()), i)
            model.train()
            likelihood.train()

            if rmse.item() < best_rmse:
                best_rmse = rmse.item()
                torch.save({'model' : model.state_dict(),
                    'i' : i,
                    'optim_state' : optim.state_dict(),
                    'objective' : loss.item(),
                    'rmse' : rmse.item(),
                    'nlpd' : nlpd.item(),
                    }, logdirname + '/best_rmse.tar')
            if nlpd.item() < best_nlpd:
                best_nlpd = nlpd.item()
                torch.save({'model' : model.state_dict(),
                    'i' : i,
                    'optim_state' : optim.state_dict(),
                    'objective' : loss.item(),
                    'rmse' : rmse.item(),
                    'nlpd' : nlpd.item(),
                    }, logdirname + '/best_nlpd.tar')

        if i % int(args['plot_interval']) == 0:
            model.eval()
            likelihood.eval()
            # evaluate everywhere
            if args['model'] == 'DiagonalGibbs':
                f_pred = model.predict(x)
            else:
                f_pred = model(x)
            # plot the mean
            mu = f_pred.mean * stdy + meany
            fig = plt.figure()
            plot_from_tensor(mu, y_grid, lons, lats, args['datestr'])
            plt.title('Predictive mean (mm), {} {}'.format(args['inference'], args['model']))
            writer.add_figure('Iteration {}: mean'.format(i+1), plt.gcf(), i, close=True)
            # plot 1 standard deviation (without observation noise)
            sigma = torch.diagonal(f_pred.covariance_matrix, dim1=-2, dim2=-1) * stdy
            fig = plt.figure()
            plot_from_tensor(sigma, y_grid, lons, lats, args['datestr'], auto_levels=True)
            plt.title('Predictive standard deviation (mm), {} {}'.format(args['inference'], args['model']))
            writer.add_figure('Iteration {}: function standard deviation'.format(i+1), plt.gcf(), i, close=True)
            # In the nonstationary case, plot the lengthscale
            if args['model'] == 'DiagonalGibbs':
                if args['inference'] == 'exact':
                    ell = model.covar_module.base_kernel.lengthscale_prior.conditional_sample(x, 
                                given=(model.train_inputs[0], torch.exp(model.log_ell_train_x)))
                else:
                    ell = model.covar_module.base_kernel.base_kernel.lengthscale_prior.conditional_sample(x,
                                given=(model.covar_module.base_kernel.inducing_points, torch.exp(model.log_ell_z)))
                directions = ['Latitudinal', 'Longitudinal']
                for d in range(2):
                    fig = plt.figure()
                    plot_from_tensor(ell[d, ...], y_grid, lons, lats, args['datestr'], auto_levels=True)
                    plt.title('{} lengthscale, {}'.format(directions[d], args['model']))
                    writer.add_figure('Iteration {}: local mean lengthscale for dim {}'.format(i+1, d+1), 
                                    plt.gcf(), i, close=True)
            model.train()
            likelihood.train()

        if loss.item() < best:
            best = loss.item()
            torch.save({'model' : model.state_dict(),
                    'i' : i,
                    'optim_state' : optim.state_dict(),
                    'objective' : loss.item(),
                    }, logdirname + '/best.tar')

        # break if not making progress
        if change < float(args['threshold']):
            break

        optim.step()

    torch.save({'model' : model.state_dict(),
            'i' : i,
            'optim_state' : optim.state_dict(),
            }, logdirname + '/final.tar')