# flake8: noqa
import os.path as osp
from statistics import mode
from basicsr.train import *
import basicsr.models

import realesrgan.archs
import realesrgan.data
import realesrgan.models
from models.realesrgan_model import RealESRGANModel
from torch.utils.tensorboard import SummaryWriter
import torch
import cv2
from tqdm import tqdm, trange
from colorama import Fore



CUDA_LAUNCH_BLOCKING=1


# rewrite training pipe
def train_pipeline_re(root_path):
    # parse options, set distributed setting, set ramdom seed
    print(root_path)
    opt, args = parse_options(root_path, is_train=True)
    opt['root_path'] = root_path
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # load resume states if necessary
    resume_state = load_resume_state(opt)
    # mkdir for experiments and logger
    if resume_state is None:
        print("resume_state===============",resume_state)
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join(opt['root_path'], 'tb_logger', opt['name']))

    # copy the yml file to the experiment root
    copy_opt_file(args.opt, opt['path']['experiments_root'])

    # WARNING: should not use get_root_logger in the above codes, including the called functions
    # Otherwise the logger will not be properly initialized
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    # initialize wandb and tb loggers
    tb_logger = init_tb_loggers(opt)

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loaders, total_epochs, total_iters = result
    total_epochs=50

    # create model
    # model = basicsr.models.build_model(opt)
    model=RealESRGANModel(opt)
    if resume_state:  # resume training
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, " f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        start_epoch = 0
        current_iter = 0

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # dataloader prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
        print("USING CPU")
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
        print("USING GPU")
    else:
        raise ValueError(f'Wrong prefetch_mode {prefetch_mode}.' "Supported ones are: None, 'cuda', 'cpu'.")

    # training
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.time()

    # load data for ConditionNet
    print('Prepare Conditional Dataset!!!!!!!!!!!!!!!!!')
    prefetcher.reset()
    train_data = prefetcher.next()
    while train_data is not None:
        model.prepare_supportset(train_data)
        train_data = prefetcher.next()
    print(model.support_pool.keys())
    model.support_pool=torch.stack(list(model.support_pool.values()))
    # model.support_pool=model.support_pool.reshape(model.support_pool.shape[0],-1,model.support_pool.shape[-2],model.support_pool.shape[-1])
    model.hr=torch.stack(list(model.hr.values()))
    # model.hr=model.hr.reshape(model.hr.shape[0],-1,model.hr.shape[-2],model.hr.shape[-1])
    # for i in model.support_pool.keys():
        # print(model.support_pool[i].shape)
    print(model.support_pool.shape)
        # print(model.kind)
    print('Complete Conditional Dataset!!!!!!!!!!!!!!!!!')

    writer = SummaryWriter('../record')

    for epoch in trange(start_epoch, total_epochs,bar_format='{l_bar}%s{bar}%s{r_bar}' % (Fore.BLUE, Fore.RESET)):
        # ConditionNet & generator trainining
        Condition_loss=model.train_Condition()
        writer.add_scalar('Loss/train', Condition_loss, epoch)

    # model.eval()
    with torch.no_grad():
        tqdm.write("model.support_pool[0].reshape(1,-1,model.support_pool[0].shape[-2],model.support_pool[0].shape[-1]).shape====={}".format(model.support_pool[0].reshape(1,-1,model.support_pool[0].shape[-2],model.support_pool[0].shape[-1]).shape))
        for i in range(model.support_pool.shape[0]):
            for j in range(model.support_pool.shape[1]//2):
                sr=model.forward(model.support_pool[i][j*2:2*j+2],model.support_pool[i][j*2:2*j+2].reshape(1,-1,model.support_pool[0].shape[-2],model.support_pool[0].shape[-1]))
                hr=model.hr[i][j*2:2*j+2]
                lr=model.support_pool[i][j*2:2*j+2]
                lr=torch.permute(lr,[0,2,3,1])
                sr=torch.permute(sr,[0,2,3,1])
                hr=torch.permute(hr,[0,2,3,1])
                lr=lr.cpu().detach().numpy()
                sr=sr.cpu().detach().numpy()
                hr=hr.cpu().detach().numpy()
                cv2.imwrite("./infer/lr/lr[{}].png".format(i*model.support_pool.shape[1]//2+2*j), lr[0]*255)
                cv2.imwrite("./infer/sr/sr[{}].png".format(i*model.support_pool.shape[1]//2+2*j), sr[0]*255)
                cv2.imwrite("./infer/hr/hr[{}].png".format(i*model.support_pool.shape[1]//2+2*j), hr[0]*255)
                cv2.imwrite("./infer/lr/lr[{}].png".format(i*model.support_pool.shape[1]//2+2*j+1), lr[1]*255)
                cv2.imwrite("./infer/sr/sr[{}].png".format(i*model.support_pool.shape[1]//2+2*j+1), sr[1]*255)
                cv2.imwrite("./infer/hr/hr[{}].png".format(i*model.support_pool.shape[1]//2+2*j+1), hr[1]*255)
        # model.forward(model.queue_lr, model.support_pool)
    #     while train_data is not None:
    #         data_timer.record()
    #
    #         current_iter += 1
    #         if current_iter > total_iters:
    #             break
    #         # update learning rate
    #         model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
    #         # training
    #         # model.feed_data(train_data)
    #         # model.optimize_parameters(current_iter)
    #         model.train_all(current_iter)
    #         # k kinds of model
    #         iter_timer.record()
    #         if current_iter == 1:
    #             # reset start time in msg_logger for more accurate eta_time
    #             # not work in resume mode
    #             msg_logger.reset_start_time()
    #         # log
    #         if current_iter % opt['logger']['print_freq'] == 0:
    #             log_vars = {'epoch': epoch, 'iter': current_iter}
    #             log_vars.update({'lrs': model.get_current_learning_rate()})
    #             log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
    #             log_vars.update(model.get_current_log())
    #             msg_logger(log_vars)
    #
    #         # save models and training states
    #         if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
    #             logger.info('Saving models and training states.')
    #             model.save(epoch, current_iter)
    #
    #         # validation
    #         if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):
    #             if len(val_loaders) > 1:
    #                 logger.warning('Multiple validation datasets are *only* supported by SRModel.')
    #             for val_loader in val_loaders:
    #                 model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])
    #
    #         data_timer.start()
    #         iter_timer.start()
    #         train_data = prefetcher.next()
    #     # end of iter
    #
    # # end of epoch
    #
    # consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    # logger.info(f'End of training. Time consumed: {consumed_time}')
    # logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    # if opt.get('val') is not None:
    #     for val_loader in val_loaders:
    #         model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])
    # if tb_logger:
    #     tb_logger.close()


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    # train_pipeline(root_path)
    train_pipeline_re(root_path)
