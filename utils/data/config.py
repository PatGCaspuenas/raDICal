def log_initial_config(logging, params, flags, paths):

    # PATHS
    path_grid = paths['grid']
    path_flow = paths['flow']
    path_flow_test = paths['flow_test']

    logging.info(f'\nPATHS \n grid: {path_grid}, flow: {path_flow}, flow_test: {path_flow_test}\n')

    # FLAGS
    flag_loss = flags['loss']
    flag_AE = flags['AE']
    flag_struct = flags['struct']
    flag_flow = flags['flow']
    flag_control = flags['control']
    flag_POD = flags['POD']
    flag_get_modal = flags['get_modal']
    flag_get_reconstruction = flags['get_reconstruction']
    flag_save_model = flags['save']['model']
    flag_save_out = flags['save']['out']
    flag_error_type = flags['error_type']

    logging.info(f'\n FLAGS \n' + \
                 f'flow type: {flag_flow}, with control? {flag_control}, error type: {flag_error_type} \n' + \
                 f'AE loss: {flag_loss}, AE type: {flag_AE}, AE structure: {flag_struct} \n' + \
                 f'POD? {flag_POD}, Modal? {flag_get_modal}, Reconstruction? {flag_get_reconstruction}, save model? {flag_save_model}, save out? {flag_save_out}\n')

    # PARAMS
    if flags['lr_static']:
        lr = params['AE']['lr']
    else:
        lr = 'variable'
    n_epochs = params['AE']['n_epochs']
    batch_size = params['AE']['batch_size']
    nr_AE = params['AE']['nr']
    beta = params['AE']['beta']

    ksize = params['AE']['ksize']
    psize = params['AE']['psize']
    ptypepool = params['AE']['ptypepool']
    nstrides = params['AE']['nstrides']
    act = params['AE']['act']
    reg_k = params['AE']['reg_k']
    reg_b = params['AE']['reg_b']

    nt_POD = params['POD']['nt']
    nr_POD = params['POD']['nr']

    n = params['flow']['n']
    m = params['flow']['m']
    Re = params['flow']['Re']
    k = params['flow']['k']


    logging.info(f'\nPARAMS \n' + \
                 f' m: {m}, n: {n}, k: {k}, Re: {Re} \n' + \
                 f' nt POD: {nt_POD}, nr POD: {nr_POD}\n' + \
                 f' nr AE: {nr_AE}, nepochs: {n_epochs}, batch size: {batch_size}, learning rate: {lr}, beta: {beta:.2E}, activation function: {act} \n' + \
                 f' kernel size: {ksize}, pool size: {psize}, pool padding: {ptypepool}, nstrides pool: {nstrides}, regularization kernel: {reg_k:.2E}, reg bias: {reg_b:.2E}\n')
