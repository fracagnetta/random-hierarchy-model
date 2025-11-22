import torch
import torch.nn as nn
import torch.nn.functional as F


def whiten(tensor, eps):	# subtract meand and divide by std along the batch dimension
    """
    Remove the tensor mean and scale by std along the batch dimension.
    
    Returns:
        Whitened tensor.
    """
    wtensor = torch.clone(tensor)
    return (wtensor-wtensor.mean(dim=0,keepdim=True))/(eps+wtensor.std(dim=0,keepdim=True))


def test( model, criterion, dataloader, device):
    """
    Test the model on data from dataloader.
    
    Returns:
        Cross-entropy loss, Classification accuracy.
    """
    model.eval()

    correct = 0
    total = 0
    loss = 0.

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            _, predictions = outputs.max(-1)

            loss += criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1)).item() * targets.numel()
            correct += predictions.eq(targets).sum().item()
            total += targets.numel()

    return loss / total, 1.0 * correct / total

# def sensitivity( model, data, transformed, device):
#     """
#     Compute the cosine similarity of the model activations when acting on original and transformed data.

#     Args:
#         model: The model (multilayer attention only).
#         data: The original input data.
#         transformed: A dictionary with several data transformations.

#     Returns:
#         Dictionary with model.num_layers+1 entries (one per hidden layer plus one for the output).
#     """
#     model.eval()

#     eps = 1e-8
#     result = {}
#     B,T,C = data.size()
#     # TODO: add batching for when B is too large

#     with torch.no_grad():

#         act_o = torch.clone(data)	# original activations
#         act_o = act_o.to(device)
#         act_t = {}                  # transformed activations
#         for k in transformed.keys():
#             act_t[k] = torch.clone(transformed[k])
#             act_t[k] = act_t[k].to(device)

#         if hasattr(model, 'token_embedding'):
#             act_o = F.linear( act_o, model.token_embedding, bias=None) *C**-.5
#             for k in transformed.keys():
#                 act_t[k] = F.linear( act_t[k], model.token_embedding, bias=None) *C**-.5
#             if hasattr(model, 'position_embedding'):
#                 act_o += model.position_embedding(torch.arange(T, device=device))
#                 for k in transformed.keys():
#                     act_t[k] += model.position_embedding(torch.arange(T, device=device))

#         for l in range(model.num_layers):

#             if hasattr(model, 'blocks'):
#                 act_o = model.blocks[l](act_o)	# compute activations on originals
#             elif hasattr(model, 'hidden'):
#                 act_o = model.hidden[l](act_o)
#             x = whiten(act_o, eps)

#             result[l] = {}
#             for k in transformed.keys():

#                 if hasattr(model, 'blocks'):
#                     act_t[k] = model.blocks[l](act_t[k])	# compute the transformed activations...
#                     x_t = whiten(act_t[k], eps)				# ...whiten over batch dimension...
#                     sensitivity = F.cosine_similarity(x, x_t, dim=2)	# ...and compute cosine_sim with originals

#                 elif hasattr(model, 'hidden'):
#                     act_t[k] = model.hidden[l](act_t[k])
#                     x_t = whiten(act_t[k], eps)				# ...whiten over batch dimension...
#                     sensitivity = F.cosine_similarity(x, x_t, dim=1)	# ...and compute cosine_sim with originals

#                 result[l][k] = sensitivity.mean(dim=0) # TODO: sum instead of mean for batching

#         x = whiten(model(data.to(device)), eps)	# same for model output
#         result[l+1] = {}
#         for k in transformed.keys():

#             x_t = whiten(model(transformed[k].to(device)), eps)
#             sensitivity = F.cosine_similarity(x, x_t, dim=1)
#             result[l+1][k] = sensitivity.mean(dim=0)

#     return result

# def sensitivity_hcnn( model, data, transformed, device):
#     """
#     Compute the cosine similarity of the model activations when acting on original and transformed data.

#     Args:
#         model: The model (multilayer attention only).
#         data: The original input data.
#         transformed: A dictionary with several data transformations.

#     Returns:
#         Dictionary with model.num_layers+1 entries (one per hidden layer plus one for the output).
#     """
#     model.eval()

#     eps = 1e-8
#     result = {}
#     B,C,T = data.size()
#     # TODO: add batching for when B is too large

#     with torch.no_grad():

#         act_o = torch.clone(data)	# original activations
#         act_o = act_o.to(device)
#         act_t = {}                  # transformed activations
#         for k in transformed.keys():
#             act_t[k] = torch.clone(transformed[k])
#             act_t[k] = act_t[k].to(device)

#         for l in range(model.num_layers):

#             result[l-0.5] = {}

#             filters = torch.clone(model.hidden[l][0].filter[:,:,-2].detach())
#             x = whiten( act_o[:,:,-2] @ filters.t(), eps)
#             # filters = torch.clone(model.hidden[l][0].filter.detach())
#             # Cout, Cin, Fs = filters.size()
#             # Np = act_o.size(-1) // Fs
#             # filters = filters.unsqueeze(-1).expand(-1, -1, -1, Np).reshape(Cout, Cin, Fs * Np).permute(2,1,0)
#             # x = torch.bmm(
#             #     act_o.permute(2,0,1),
#             #     filters.cpu()
#             # ).permute(1,2,0)
#             # x = whiten( x, eps)

#             for k in transformed.keys():
#                 x_t = whiten( act_t[k][:,:,-2] @ filters.t(), eps)
#                 # x_t = torch.bmm(
#                 #     act_t[k].permute(2,0,1),
#                 #     filters
#                 # ).permute(1,2,0)
#                 # x_t = whiten(x_t, eps)
#                 sensitivity = F.cosine_similarity(x, x_t, dim=1)	# ...and compute cosine_sim with originals
#                 result[l-0.5][k] = sensitivity.mean(dim=0) # TODO: sum instead of mean for batching

#             act_o = model.hidden[l](act_o)
#             x = whiten(act_o, eps)

#             result[l] = {}

#             for k in transformed.keys():

#                 act_t[k] = model.hidden[l](act_t[k])
#                 x_t = whiten(act_t[k], eps)				# ...whiten over batch dimension...
#                 sensitivity = F.cosine_similarity(x, x_t, dim=1)	# ...and compute cosine_sim with originals
#                 result[l][k] = sensitivity.mean(dim=0) # TODO: sum instead of mean for batching

#         x = whiten(model(data.to(device)), eps)	# same for model output
#         result[l+1] = {}
#         for k in transformed.keys():

#             x_t = whiten(model(transformed[k].to(device)), eps)
#             sensitivity = F.cosine_similarity(x, x_t, dim=1)
#             result[l+1][k] = sensitivity.mean(dim=0)

#     return result