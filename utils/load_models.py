import torch
from mm_fit.model.conv_ae import ConvAutoencoder
from mm_fit.model.multimodal_ae import MultimodalAutoencoder
from mm_fit.model.multimodal_ar import MultimodalFcClassifier

def load_models(ae_layers = 3, ae_hidden_units = 1000, embedding_units = 1000, ae_dropout = 0.0,
                model_wp = "output/mmfit_demo_1610541736_checkpoint_0.pth", device = 'cpu',
                num_classes = 11, layers = 2, hidden_units = 1000, dropout = 0.0, window_stride = 0.2,
                window_length = 5, skeleton_sampling_rate = 30, target_sensor_sampling_rate = 50, 
                modalities = ["sw_l_acc", "sw_l_gyr", "sw_r_acc", "sw_r_gyr", "eb_acc", "eb_gyr", "sp_acc", "sp_gyr", "skel"]):

    model_modalities = correct_names(modalities)

    window_stride = int(window_stride * skeleton_sampling_rate)
    skeleton_window_length = int(window_length * skeleton_sampling_rate)
    sensor_window_length = int(window_length * target_sensor_sampling_rate)

    sw_l_acc_model = ConvAutoencoder(input_size=sensor_window_length, input_ch=3, dim=1, layers=3, grouped=[3, 3, 1],
                                    kernel_size=11, kernel_stride=2, return_embeddings=True).to(device, non_blocking=True)

    sw_l_gyr_model = ConvAutoencoder(input_size=sensor_window_length, input_ch=3, dim=1, layers=3, grouped=[1, 1, 1],
                                    kernel_size=3, kernel_stride=2, return_embeddings=True).to(device, non_blocking=True)

    sw_r_acc_model = ConvAutoencoder(input_size=sensor_window_length, input_ch=3, dim=1, layers=3, grouped=[3, 3, 1],
                                    kernel_size=11, kernel_stride=2, return_embeddings=True).to(device, non_blocking=True)

    sw_r_gyr_model = ConvAutoencoder(input_size=sensor_window_length, input_ch=3, dim=1, layers=3, grouped=[1, 1, 1],
                                    kernel_size=3, kernel_stride=2, return_embeddings=True).to(device, non_blocking=True)

    eb_acc_model = ConvAutoencoder(input_size=sensor_window_length, input_ch=3, dim=1, layers=3, grouped=[3, 3, 1],
                                kernel_size=11, kernel_stride=2, return_embeddings=True).to(device, non_blocking=True)

    eb_gyr_model = ConvAutoencoder(input_size=sensor_window_length, input_ch=3, dim=1, layers=3, grouped=[1, 1, 1],
                                kernel_size=3, kernel_stride=2, return_embeddings=True).to(device, non_blocking=True)

    sp_acc_model = ConvAutoencoder(input_size=sensor_window_length, input_ch=3, dim=1, layers=3, grouped=[1, 1, 1],
                                kernel_size=11, kernel_stride=2, return_embeddings=True).to(device, non_blocking=True)

    sp_gyr_model = ConvAutoencoder(input_size=sensor_window_length, input_ch=3, dim=1, layers=3, grouped=[1, 1, 1],
                                kernel_size=3, kernel_stride=2, return_embeddings=True).to(device, non_blocking=True)

    skel_model = ConvAutoencoder(input_size=(skeleton_window_length, 16), input_ch=3, dim=2, layers=3, grouped=[3, 3, 1],
                                kernel_size=11, kernel_stride=(2, 1), return_embeddings=True).to(device, non_blocking=True)

    multimodal_ae_f_in = 4800
    multimodal_ae_model = MultimodalAutoencoder(f_in=multimodal_ae_f_in, sw_l_acc=sw_l_acc_model, sw_l_gyr=sw_l_gyr_model,
                                                sw_r_acc=sw_r_acc_model, sw_r_gyr=sw_r_gyr_model, eb_acc=eb_acc_model,
                                                eb_gyr=eb_gyr_model, sp_acc=sp_acc_model, sp_gyr=sp_gyr_model,
                                                skel=skel_model, layers=ae_layers, hidden_units=ae_hidden_units,
                                                f_embedding=embedding_units, dropout=ae_dropout,
                                                return_embeddings=True).to(device, non_blocking=True)

    model = MultimodalFcClassifier(f_in=embedding_units, num_classes=num_classes, multimodal_ae_model=multimodal_ae_model,
                                layers=layers, hidden_units=hidden_units,
                                dropout=dropout).to(device, non_blocking=True)

    model_params = torch.load(model_wp, map_location=device)
    model.load_state_dict(model_params['model_state_dict'])

    models = {}
    for name, (model_name, child) in zip(modalities, model.multimodal_ae_model.named_children()):
        if model_name in model_modalities:
            models[name] = child
    return models

def correct_names(modalities):
    model_names = []
    for name in modalities:
        if name in ['eb_l_acc', 'eb_r_acc']:
            model_names.append('eb_acc')
        elif name in ['eb_l_gyr', 'eb_r_gyr']:
            model_names.append('eb_gyr')
        elif name in ['sp_l_acc', 'sp_r_acc']:
            model_names.append('sp_acc')
        elif name in ['sp_l_gyr', 'sp_r_gyr']:
            model_names.append('sp_gyr')
        elif name in['pose_3d']:
            model_names.append('skel')
        else:
            model_names.append(name)
    return model_names
    