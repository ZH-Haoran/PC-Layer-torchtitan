import json
import matplotlib.pyplot as plt

path_no_pc = ''

path_pc = ''

with open(f'{path_pc}', 'r') as f:
    grad_kappa_pc_dict = json.load(f)

with open(f'{path_no_pc}', 'r') as f:
    grad_kappa_no_pc_dict = json.load(f)


with open(f'{path_pc}', 'r') as f:
    grad_norm_pc_dict = json.load(f)

with open(f'{path_no_pc}', 'r') as f:
    grad_norm_no_pc_dict = json.load(f)


for name in grad_kappa_pc_dict.keys():

    # plot the list of kappa
    plt.figure(figsize=(10, 5))
    plt.title(f"Kappa of {name}")
    plt.plot(grad_kappa_pc_dict[name], label = "kappa of W")
    # plt.plot(pc_kappa_normalized_dict[name], label = "kappa of W / ||W||_op")
    plt.plot(grad_kappa_no_pc_dict[name], label = "kappa of W")
    plt.legend()
    plt.savefig(f"figures/{job_config.metrics.wandb_comment}/kappa_{name}.png")
    plt.close()



    # plot the list of kappa
    plt.figure(figsize=(10, 5))
    plt.title(f"Kappa of {name}")
    plt.plot(grad_norm_pc_dict[name], label = "kappa of W")
    # plt.plot(pc_kappa_normalized_dict[name], label = "kappa of W / ||W||_op")
    plt.plot(grad_norm_no_pc_dict[name], label = "kappa of W")
    plt.legend()
    plt.savefig(f"figures/{job_config.metrics.wandb_comment}/kappa_{name}.png")
    plt.close()



