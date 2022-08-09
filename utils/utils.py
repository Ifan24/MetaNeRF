from datetime import datetime
import os
import json


def make_dir(args):
    """make a directory in checkpoint_path with name as current time and sava the args as json file

    Args:
        args (_type_): parser args
    return:
        the new checkpoint path
    """
    
    # dd/mm/YY H:M:S
    now = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    checkpoint_path = f"{args.checkpoint_path}/{now}"
    os.makedirs(checkpoint_path, exist_ok=True)
    print(f"make directory {checkpoint_path}")
    checkpoint_path
    
    with open(f'{checkpoint_path}/config.json', 'w') as fp:
        json.dump(vars(args), fp, indent=4)
        
    return checkpoint_path