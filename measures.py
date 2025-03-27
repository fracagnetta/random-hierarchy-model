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


def test( model, dataloader, device):
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
            _, predictions = outputs.max(1)

            loss += F.cross_entropy(outputs, targets, reduction='sum').item()
            correct += predictions.eq(targets).sum().item()
            total += targets.size(0)

    return loss / total, 1.0 * correct / total


def predict( model, data, device):
    """

    Args:
        model: The model (multilayer attention only).
        data: The original input data.

    Returns:

    """
    model.eval()
    with torch.no_grad():

        outputs = model(data.to(device))
        _, predictions = outputs.max(1)

    return predictions


def sensitivity( model, data, transformed, device):
    """
    Compute the cosine similarity of the model activations when acting on original and transformed data.

    Args:
        model: The model (multilayer attention only).
        data: The original input data.
        transformed: A dictionary with several data transformations.

    Returns:
        Dictionary with model.num_layers+1 entries (one per hidden layer plus one for the output).
    """
    model.eval()

    eps = 1e-8
    result = {}
    B,T,C = data.size()
    # TODO: add batching for when B is too large

    with torch.no_grad():

        act_o = torch.clone(data)	# original activations
        act_o = act_o.to(device)
        act_t = {}                  # transformed activations
        for k in transformed.keys():
            act_t[k] = torch.clone(transformed[k])
            act_t[k] = act_t[k].to(device)

        if hasattr(model, 'token_embedding'):
            act_o = F.linear( act_o, model.token_embedding, bias=None) *C**-.5
            for k in transformed.keys():
                act_t[k] = F.linear( act_t[k], model.token_embedding, bias=None) *C**-.5
            if hasattr(model, 'position_embedding'):
                act_o += model.position_embedding(torch.arange(T, device=device))
                for k in transformed.keys():
                    act_t[k] += model.position_embedding(torch.arange(T, device=device))

        for l in range(model.num_layers):

            if hasattr(model, 'blocks'):
                act_o = model.blocks[l](act_o)	# compute activations on originals
            elif hasattr(model, 'hidden'):
                act_o = model.hidden[l](act_o)
            x = whiten(act_o, eps)

            result[l] = {}
            for k in transformed.keys():

                if hasattr(model, 'blocks'):
                    act_t[k] = model.blocks[l](act_t[k])	# compute the transformed activations...
                    x_t = whiten(act_t[k], eps)				# ...whiten over batch dimension...
                    sensitivity = F.cosine_similarity(x, x_t, dim=2)	# ...and compute cosine_sim with originals

                elif hasattr(model, 'hidden'):
                    act_t[k] = model.hidden[l](act_t[k])
                    x_t = whiten(act_t[k], eps)				# ...whiten over batch dimension...
                    sensitivity = F.cosine_similarity(x, x_t, dim=1)	# ...and compute cosine_sim with originals

                result[l][k] = sensitivity.mean(dim=0) # TODO: sum instead of mean for batching

        x = whiten(model(data.to(device)), eps)	# same for model output
        result[l+1] = {}
        for k in transformed.keys():

            x_t = whiten(model(transformed[k].to(device)), eps)
            sensitivity = F.cosine_similarity(x, x_t, dim=1)
            result[l+1][k] = sensitivity.mean(dim=0)

    return result

def check_rules_clean( samples, rules):
    """
    Check if clean samples are consistent the production rules of the grammar.

    Args:
        samples: A tensor of size (B,d,v).
        rules: A dictionary of production rules (tensors of size (v,m,s)) with keys 0,...,L-1.
               rules[l]_{i,j} = s-tuple produced by the j-th rule emanating from i.

    Returns:
        level_accuracy: A dictionary of accuracies (tensors of size (s**l)) with keys 0,...,L-1.
                        level_accuracy[l]_{i} = fraction of compatible rules in position i
        rules_frequencies: A dictionary of rules occurrencies (tensors of size (v*m)) with keys 0,...,L-1.
    """
    B, d, v = samples.shape
    samples = F.one_hot(samples.argmax(dim=2),num_classes=v)
    L = len(rules)
    v, m, s = rules[0].shape
    upwd_messages = samples.permute(2,0,1).flatten(start_dim=1) # initial upward messages, size (v,B*s**L)

    level_accuracy = {}
    rules_frequencies = {}

    for l in range(L-1,-1,-1):

        rules_flat = rules[l].reshape(-1, s)
        prob_rules = upwd_messages.reshape(v,-1,s).transpose(1,2)
        prob_rules = prob_rules[rules_flat, torch.arange(s)].squeeze().prod(1)
        prob_rules = prob_rules.reshape(v,m,-1)
        upwd_messages = prob_rules.sum(dim=1) # messages for the next level, size (v, B*s**(L-l))

        prob_rules = prob_rules.reshape(v,m,B,-1)
        level_accuracy[l] = prob_rules.sum(dim=(0,1,2)) / B # size (s**l)
        rules_frequencies[l] = prob_rules.sum(dim=(2,3)).flatten() # size (v*m)
    
    return level_accuracy, rules_frequencies

def test_rules( rules, model, model_name, dataloader, device):
    """
    Test the compatibility of predictions with the rules.
    
    Returns:
        
    """
    model.eval()
    L = len(rules)
    v, m, s = rules[0].shape
    rules_accuracy = {}
    rules_frequency = {}
    num_batches = 0
    for l in range(L):
        rules_accuracy[l] = torch.zeros(s**l, device=device)
        rules_frequency[l] = torch.zeros(v*m, device=device)
   
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            _, predictions = outputs.max(1) # TODO: sample predictions instead of taking the max?
            
            if 'fcn' in model_name:
                inputs = inputs.reshape(B,-1,v)
                inputs = torch.cat((inputs,F.one_hot(predictions, num_classes=v).view(-1,1,v)),dim=1)

            elif 'transformer' in model_name:
                inputs[:,-1,:] = F.one_hot(predictions, num_classes=v)

            elif 'hcnn' in model_name:
                inputs = inputs.transpose(1,2)
                inputs[:,-1,:] = F.one_hot(predictions, num_classes=v)
            
            r_acc, r_freq = check_rules_clean(inputs, rules)
            for l in range(L):
                rules_accuracy[l] += r_acc[l]
                rules_frequency[l] += r_freq[l]
            num_batches += 1
    
    for l in range(L):
        rules_accuracy[l] /= num_batches

    return rules_accuracy, rules_frequency