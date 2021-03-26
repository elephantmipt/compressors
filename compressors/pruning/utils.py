from torch.nn.utils import prune


def create_identity_mask(model, tensor_name="weight"):
    for _name, module in model.named_modules():
        try:
            prune.identity(module, tensor_name)
        except Exception as e:
            print(e)
            pass
