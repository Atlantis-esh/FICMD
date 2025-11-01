import torch
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import SpectralClustering
import config

# This script is adapted from the user-provided Ours1_test_ProtoAtt.py file.
# It has been refactored for clarity and integration with the current project structure.

class MinNormSolver:
    MAX_ITER = 250
    STOP_CRIT = 1e-5

    @staticmethod
    def _min_norm_element_from2(v1v1, v1v2, v2v2):
        if v1v2 >= v1v1:
            return 0.999, v1v1
        if v1v2 >= v2v2:
            return 0.001, v2v2
        gamma = -1.0 * ((v1v2 - v2v2) / (v1v1 + v2v2 - 2 * v1v2))
        cost = v2v2 + gamma * (v1v2 - v2v2)
        return gamma, cost

    @staticmethod
    def _min_norm_2d(vecs, dps):
        dmin = 1e8
        sol = None
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                if (i, j) not in dps:
                    dps[(i, j)] = sum(torch.mul(v, w).sum() for v, w in zip(vecs[i], vecs[j]) if v is not None and w is not None).item()
                    dps[(j, i)] = dps[(i, j)]
                if (i, i) not in dps:
                    dps[(i, i)] = sum(torch.mul(v, v).sum() for v in vecs[i] if v is not None).item()
                if (j, j) not in dps:
                    dps[(j, j)] = sum(torch.mul(v, v).sum() for v in vecs[j] if v is not None).item()
                c, d = MinNormSolver._min_norm_element_from2(dps[(i, i)], dps[(i, j)], dps[(j, j)])
                if d < dmin:
                    dmin = d
                    sol = [(i, j), c, d]
        return sol, dps

    @staticmethod
    def _projection2simplex(y):
        m = len(y)
        sorted_y = np.flip(np.sort(y), axis=0)
        tmpsum = 0.0
        tmax_f = (np.sum(y) - 1.0) / m
        for i in range(m - 1):
            tmpsum += sorted_y[i]
            tmax = (tmpsum - 1) / (i + 1.0)
            if tmax > sorted_y[i + 1]:
                tmax_f = tmax
                break
        return np.maximum(y - tmax_f, 0)

    @staticmethod
    def _next_point(cur_val, grad, n):
        proj_grad = grad - (np.sum(grad) / n)
        tm1 = -1.0 * cur_val[proj_grad < 0] / proj_grad[proj_grad < 0]
        tm2 = (1.0 - cur_val[proj_grad > 0]) / (proj_grad[proj_grad > 0])
        
        t = 1.0
        if len(tm1[tm1 > 1e-7]) > 0:
            t = np.min(tm1[tm1 > 1e-7])
        if len(tm2[tm2 > 1e-7]) > 0:
            t = min(t, np.min(tm2[tm2 > 1e-7]))

        next_point = cur_val + t * proj_grad
        next_point = MinNormSolver._projection2simplex(next_point)
        return next_point

    @staticmethod
    def find_min_norm_element(vecs):
        valid_vecs = [v for v in vecs if v is not None and any(g is not None for g in v)]
        if not valid_vecs:
            return np.ones(len(vecs)) / len(vecs), 0.0
        
        dps, n = {}, len(valid_vecs)
        init_sol, dps = MinNormSolver._min_norm_2d(valid_vecs, dps)
        
        if init_sol is None:
            return np.ones(len(vecs)) / len(vecs), 0.0

        sol_vec = np.zeros(n)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            full_sol = np.zeros(len(vecs))
            valid_indices = [i for i, v in enumerate(vecs) if v is not None and any(g is not None for g in v)]
            for i, val in enumerate(sol_vec):
                full_sol[valid_indices[i]] = val
            return full_sol, init_sol[2]
        
        grad_mat = np.array([[dps[(i, j)] for j in range(n)] for i in range(n)])
        
        for _ in range(MinNormSolver.MAX_ITER):
            grad_dir = -np.dot(grad_mat, sol_vec)
            new_point = MinNormSolver._next_point(sol_vec, grad_dir, n)
            v1v1 = np.dot(sol_vec, np.dot(grad_mat, sol_vec))
            v1v2 = np.dot(sol_vec, np.dot(grad_mat, new_point))
            v2v2 = np.dot(new_point, np.dot(grad_mat, new_point))
            
            nc, nd = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc * sol_vec + (1 - nc) * new_point
            change = new_sol_vec - sol_vec
            if np.sum(np.abs(change)) < MinNormSolver.STOP_CRIT:
                break
            sol_vec = new_sol_vec

        full_sol = np.zeros(len(vecs))
        valid_indices = [i for i, v in enumerate(vecs) if v is not None and any(g is not None for g in v)]
        for i, val in enumerate(sol_vec):
            full_sol[valid_indices[i]] = val
        return full_sol, nd


def gradient_normalizers(grads, losses, normalization_type):
    gn = {}
    if normalization_type == 'l2':
        for t, g in enumerate(grads):
            if g is None: gn[t] = 0.0; continue
            valid_grads = [gr for gr in g if gr is not None]
            gn[t] = torch.sqrt(sum(gr.pow(2).sum() for gr in valid_grads)).item() if valid_grads else 0.0
    elif normalization_type == 'loss':
        for t in range(len(grads)): gn[t] = losses[t] if t < len(losses) else 0.0
    elif normalization_type == 'loss+':
        for t, g in enumerate(grads):
            if g is None: gn[t] = losses[t] if t < len(losses) else 0.0; continue
            valid_grads = [gr for gr in g if gr is not None]
            gn[t] = (losses[t] * torch.sqrt(sum(gr.pow(2).sum() for gr in valid_grads))).item() if valid_grads else (losses[t] if t < len(losses) else 0.0)
    elif normalization_type == 'none':
        for t in range(len(grads)): gn[t] = 1.0
    else:
        raise ValueError(f"Unknown normalization type: {normalization_type}")
    return gn

def get_grads(model, loss):
    """Computes and returns the gradients of a model's parameters for a given loss."""
    model.zero_grad()
    loss.backward(retain_graph=True)
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.data.clone())
        else:
            grads.append(None)
    return grads

def _cluster_gradients(grads_list, n_groups):
    """Spectral clustering on a list of gradient vectors, robust version."""
    # Robust stack and normalization
    grad_matrix = torch.stack([
        torch.cat([g.flatten().cpu() for g in grad_vec if g is not None])
        for grad_vec in grads_list
    ])
    grad_norms = grad_matrix.norm(dim=1, keepdim=True)
    grad_matrix = grad_matrix / (grad_norms + 1e-8)
    # Cosine similarity
    similarity_matrix = torch.mm(grad_matrix, grad_matrix.t())
    similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2
    similarity_matrix = torch.clamp(similarity_matrix, -1, 1)
    # Ensure no NaN
    if torch.isnan(similarity_matrix).any():
        print('Warning: similarity_matrix contains NaN, using uniform grouping.')
        n = grad_matrix.shape[0]
        groups = [[] for _ in range(n_groups)]
        indices = np.arange(n)
        for i, idx in enumerate(indices):
            groups[i % n_groups].append(idx)
        return [g for g in groups if len(g) > 0]
    try:
        clustering = SpectralClustering(n_clusters=n_groups, affinity='precomputed', random_state=42)
        labels = clustering.fit_predict(similarity_matrix.cpu().numpy())
        groups = [[] for _ in range(n_groups)]
        for i, label in enumerate(labels):
            groups[label].append(i)
        # Check for empty groups
        if not all(groups):
            print('Spectral clustering produced empty group, using uniform grouping.')
            n = grad_matrix.shape[0]
            groups = [[] for _ in range(n_groups)]
            indices = np.arange(n)
            for i, idx in enumerate(indices):
                groups[i % n_groups].append(idx)
        return [g for g in groups if len(g) > 0]
    except Exception as e:
        print(f'Spectral clustering failed: {e}. Using uniform grouping.')
        n = grad_matrix.shape[0]
        groups = [[] for _ in range(n_groups)]
        indices = np.arange(n)
        for i, idx in enumerate(indices):
            groups[i % n_groups].append(idx)
        return [g for g in groups if len(g) > 0]


def balanced_gradient_update(model, optimizer, batch, loss_fn, n_groups, metadata):
    """
    Performs the full, combined gradient update as per Entry Point 3.
    1. Computes class-level and sample-level gradients.
    2. Clusters them using spectral clustering.
    3. Solves the min-norm problem for both sets of groups.
    4. Combines the weighted gradients and updates the model.
    """
    inputs, labels, modalities = batch['image'], batch['label'], batch['modality']
    
    # --- 1. Class-level balancing ---
    class_grads, class_losses = [], []
    unique_classes = torch.unique(labels)
    for c in unique_classes:
        mask = (labels == c)
        if not mask.any(): continue
        
        class_batch = {k: (v[mask] if torch.is_tensor(v) else v) for k, v in batch.items()}
        # print(f'class_batch={class_batch}')
        class_loss = loss_fn(model(class_batch), class_batch)
        class_grads.append(get_grads(model, class_loss))
        class_losses.append(class_loss.item())
    class_weights = solve_min_norm_problem(class_grads, class_losses) if class_grads else None
    print(f'class_weights={class_weights}=====================')
    # --- 2. Sample-level balancing ---
    sample_grads, sample_losses = [], []
    for i in range(len(labels)):
        sample_batch = {k: (v[i:i+1] if torch.is_tensor(v) else v) for k, v in batch.items()}
        sample_loss = loss_fn(model(sample_batch), sample_batch)
        sample_grads.append(get_grads(model, sample_loss))
        sample_losses.append(sample_loss.item())
    sample_groups = _cluster_gradients(sample_grads, n_groups)
    print(f'sample_groups={sample_groups}----------------------')

    grouped_sample_grads, grouped_sample_losses = [], []
    for group_indices in sample_groups:
        if not group_indices: continue
        
        group_loss = torch.mean(torch.stack([torch.tensor(sample_losses[i]) for i in group_indices]))
        
        # Average gradients within the group
        avg_grad = []
        num_params = len(sample_grads[0])
        for p_idx in range(num_params):
            param_grads = [sample_grads[s_idx][p_idx] for s_idx in group_indices if sample_grads[s_idx][p_idx] is not None]
            if param_grads:
                avg_grad.append(torch.stack(param_grads).mean(dim=0))
            else:
                avg_grad.append(None)
        
        grouped_sample_grads.append(avg_grad)
        grouped_sample_losses.append(group_loss.item())

    sample_weights = solve_min_norm_problem(grouped_sample_grads, grouped_sample_losses) if grouped_sample_grads else None
    
    # --- 3. Combine and apply gradients ---
    optimizer.zero_grad()
    
    final_grad = [torch.zeros_like(p) for p in model.parameters() if p.requires_grad]
    param_map = [p for p in model.parameters() if p.requires_grad]


    # Apply class-level gradients
    if class_weights is not None:
        for i, weight in enumerate(class_weights):
            if i >= len(class_grads) or class_grads[i] is None: continue
            for j, g in enumerate(class_grads[i]):
                if g is None: continue
                # Find the corresponding parameter in final_grad
                for k, p in enumerate(model.parameters()):
                     if p.grad is not None and id(p) == id(param_map[j]):
                        final_grad[k] += (1 - config.W_MIX) * weight * g
                        break

    # Apply sample-level gradients
    if sample_weights is not None:
        for i, weight in enumerate(sample_weights):
            if i >= len(grouped_sample_grads) or grouped_sample_grads[i] is None: continue
            for j, g in enumerate(grouped_sample_grads[i]):
                if g is None: continue
                # Find the corresponding parameter in final_grad
                for k, p in enumerate(model.parameters()):
                     if p.grad is not None and id(p) == id(param_map[j]):
                        final_grad[k] += config.W_MIX * weight * g
                        break
    
    # Set the computed gradients and step the optimizer
    grad_applied = False
    for p, g in zip(model.parameters(), final_grad):
        if p.requires_grad:
            p.grad = g
            grad_applied = True
    
    if grad_applied:
        optimizer.step()


def solve_min_norm_problem(grads, losses, normalization_type='l2'):
    gn = gradient_normalizers(grads, losses, normalization_type)
    normalized_grads = []
    for i, grad in enumerate(grads):
        if grad is None:
            normalized_grads.append(None)
            continue
        normalized_grad = []
        for g in grad:
            if g is None:
                normalized_grad.append(None)
                continue
            if i in gn and gn[i] > 0:
                normalized_grad.append(g / gn[i])
            else:
                normalized_grad.append(g)
        normalized_grads.append(normalized_grad)
    sol, _ = MinNormSolver.find_min_norm_element(normalized_grads)
    return sol

# ==== BEGIN: Dynamic gradient optimization functions (aligned with Ours1_test_ProtoAtt.py) ====
import torch
import numpy as np
from sklearn.cluster import SpectralClustering
import torch.nn.functional as F

def build_graph_from_similarity(similarity_matrix, threshold=0.2):
    mask = (similarity_matrix > threshold)
    adj_matrix = torch.zeros_like(similarity_matrix)
    mask.fill_diagonal_(False)
    adj_matrix[mask] = 1
    return adj_matrix

def spectral_clustering(adj_matrix, n_groups):
    adj_matrix = adj_matrix + 1e-8 * torch.eye(adj_matrix.shape[0], device=adj_matrix.device)
    adj_matrix = (adj_matrix + adj_matrix.T) / 2
    try:
        clustering = SpectralClustering(
            n_clusters=n_groups,
            affinity='precomputed',
            random_state=42
        )
        labels = clustering.fit_predict(adj_matrix.cpu().numpy())
        groups = [[] for _ in range(n_groups)]
        for i, label in enumerate(labels):
            groups[label].append(i)
        return [g for g in groups if len(g) > 0]
    except Exception as e:
        print(f"Spectral clustering failed, using uniform grouping: {e}")
        groups = [[] for _ in range(n_groups)]
        indices = np.arange(adj_matrix.shape[0])
        for i, idx in enumerate(indices):
            groups[i % n_groups].append(idx)
        return [g for g in groups if len(g) > 0]

def compute_class_grads_in_batch(model, batch, num_classes, device):
    # Returns: {class_id: gradient_vector}
    grads_per_class = {}
    images = batch['image']
    targets = batch['label']
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    modalities = batch['modality']
    img_modality = batch['img_modality']
    batch_classes = torch.unique(targets)
    for c in batch_classes:
        mask = targets == c
        if mask.sum() == 0:
            continue
        model.zero_grad()
        logits = model.get_logits(images[mask], input_ids[mask], attention_mask[mask], modalities[mask], img_modality[mask])
        loss = F.cross_entropy(logits, targets[mask])
        loss.backward()
        grads = []
        for param in model.parameters():
            if param.grad is not None:
                grad = param.grad.data.clone().flatten().cpu()
                grad_norm = torch.norm(grad)
                if grad_norm > 1e-8:
                    grad = grad / grad_norm
                grads.append(grad)
        grad_vec = torch.cat(grads)
        grads_per_class[int(c)] = grad_vec
        del grads, grad_vec, logits, loss
        torch.cuda.empty_cache()
    return grads_per_class

def gradient_balanced_grouping(model, batch, num_classes, n_groups, device):
    mode = getattr(config, 'CLASS_GROUPING_MODE', 'batch_group')
    if mode == 'batch_group':
        # Direct grouping by class, return [[c1],[c2],...]
        grads_per_class = compute_class_grads_in_batch(model, batch, num_classes, device)
        groups = [[c] for c in grads_per_class.keys()]
        print(f"Class grouping result: {groups}")
        return groups
    elif mode == 'batch_mean':
        # First compute average gradient per class, then spectral clustering into n groups
        grads_per_class = compute_class_grads_in_batch(model, batch, num_classes, device)
        valid_classes = list(grads_per_class.keys())
        if not valid_classes:
            return []
        grad_matrix = torch.stack([grads_per_class[c] for c in valid_classes])
        similarity_matrix = torch.mm(grad_matrix, grad_matrix.t())
        similarity_matrix = torch.clamp(similarity_matrix, -1.0, 1.0)
        for i in range(len(valid_classes)):
            similarity_matrix[i, i] = 1.0
        adj_matrix = build_graph_from_similarity(similarity_matrix)
        # Spectral clustering grouping, return class ids in groups
        cluster_groups = spectral_clustering(adj_matrix, n_groups)
        # Convert to class ids
        groups = [[valid_classes[i] for i in group] for group in cluster_groups]
        print(f"Class grouping result: {groups}")
        return groups
    else:
        # # Original full dataset approach
        # similarity_matrix = compute_gradient_similarity_matrix(model, batch, num_classes, device, mode='all_data')
        # adj_matrix = build_graph_from_similarity(similarity_matrix)
        # groups = spectral_clustering(adj_matrix, n_groups)
        # print(f"Class grouping result: {groups}")
        # return groups
        print('errors!!!!!!!!!!!!!!!')

def get_group_grads(model, batch, class_groups, device, loss_fn=None):
    group_grads = []
    group_losses = []
    images = batch['image'].to(device)
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    modalities = batch['modality'].to(device)
    img_modality = batch['img_modality'].to(device)
    targets = batch['label'].to(device)
    for group in class_groups:
        group_mask = torch.zeros_like(targets, dtype=torch.bool)
        for cls in group:
            group_mask |= (targets == cls)
        if not group_mask.any():
            zero_grads = []
            for param in model.parameters():
                if param.requires_grad:
                    zero_grads.append(torch.zeros_like(param.data))
                else:
                    zero_grads.append(None)
            group_grads.append(zero_grads)
            group_losses.append(0.0)
            continue
        model.zero_grad()
        # Use loss_fn to fuse multi-branch losses
        if loss_fn is not None:
            sub_batch = {k: v[group_mask] if isinstance(v, torch.Tensor) and v.shape[0]==targets.shape[0] else v for k,v in batch.items()}
            outputs = model(sub_batch)
            loss = loss_fn(outputs, sub_batch)
        else:
            logits = model.get_logits(images[group_mask], input_ids[group_mask], attention_mask[group_mask], modalities[group_mask], img_modality[group_mask])
            loss = F.cross_entropy(logits, targets[group_mask])
        loss.backward()
        grads = []
        for param in model.parameters():
            if param.grad is not None:
                grads.append(param.grad.data.clone())
            else:
                grads.append(None)
        group_grads.append(grads)
        group_losses.append(loss.item())
    return group_grads, group_losses

def get_individual_sample_grads(model, batch, device, loss_fn=None):
    sample_grads = []
    images = batch['image'].to(device)
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    modalities = batch['modality'].to(device)
    img_modality = batch['img_modality'].to(device)
    targets = batch['label'].to(device)
    for i in range(len(images)):
        model.zero_grad()
        if loss_fn is not None:
            sub_batch = {k: v[i:i+1] if isinstance(v, torch.Tensor) and v.shape[0]==images.shape[0] else v for k,v in batch.items()}
            outputs = model(sub_batch)
            loss = loss_fn(outputs, sub_batch)
        else:
            logits = model.get_logits(images[i:i+1], input_ids[i:i+1], attention_mask[i:i+1], modalities[i:i+1], img_modality[i:i+1])
            loss = F.cross_entropy(logits, targets[i:i+1])
        loss.backward()
        grads = []
        for param in model.parameters():
            if param.grad is not None:
                grads.append(param.grad.data.clone().flatten())
        if grads:
            grad_vec = torch.cat(grads).detach().cpu()
            sample_grads.append(grad_vec)
    return sample_grads

def compute_sample_similarity_matrix(sample_grads):
    n_samples = len(sample_grads)
    similarity_matrix = torch.zeros((n_samples, n_samples))
    grad_matrix = torch.stack(sample_grads)
    grad_norms = torch.norm(grad_matrix, dim=1, keepdim=True)
    grad_matrix = grad_matrix / (grad_norms + 1e-8)
    similarity_matrix = torch.mm(grad_matrix, grad_matrix.t())
    similarity_matrix.fill_diagonal_(1.0)
    return similarity_matrix

def cluster_sample_gradients(sample_grads, n_groups):
    try:
        similarity_matrix = compute_sample_similarity_matrix(sample_grads)
        adj_matrix = build_graph_from_similarity(similarity_matrix, threshold=0.2)
        groups = spectral_clustering(adj_matrix, n_groups)
        sample_groups = groups
        if not all(sample_groups):
            print("Spectral clustering produced empty groups, using uniform grouping")
            n_samples = len(sample_grads)
            sample_groups = [[] for _ in range(n_groups)]
            indices = np.random.permutation(n_samples)
            for i, idx in enumerate(indices):
                sample_groups[i % n_groups].append(idx)
    except Exception as e:
        print(f"Sample clustering failed: {e}, using random grouping")
        n_samples = len(sample_grads)
        sample_groups = [[] for _ in range(n_groups)]
        indices = np.random.permutation(n_samples)
        for i, idx in enumerate(indices):
            sample_groups[i % n_groups].append(idx)
    return sample_groups

def update_model_with_sample_level_mgda(model, batch, n_groups, device, loss_fn=None):
    sample_grads = get_individual_sample_grads(model, batch, device, loss_fn=loss_fn)
    if not sample_grads:
        return None, None
    sample_groups = cluster_sample_gradients(sample_grads, n_groups)
    group_grads = []
    group_losses = []
    images = batch['image'].to(device)
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    modalities = batch['modality'].to(device)
    img_modality = batch['img_modality'].to(device)
    targets = batch['label'].to(device)
    for group in sample_groups:
        if not group:
            continue
        group_idx = torch.tensor(group, dtype=torch.long, device=device)
        group_images = images[group_idx]
        group_input_ids = input_ids[group_idx]
        group_attention_mask = attention_mask[group_idx]
        group_modalities = modalities[group_idx]
        group_img_modality = img_modality[group_idx]
        group_targets = targets[group_idx]
        model.zero_grad()
        if loss_fn is not None:
            sub_batch = {k: v[group_idx] if isinstance(v, torch.Tensor) and v.shape[0]==images.shape[0] else v for k,v in batch.items()}
            outputs = model(sub_batch)
            loss = loss_fn(outputs, sub_batch)
        else:
            logits = model.get_logits(group_images, group_input_ids, group_attention_mask, group_modalities, group_img_modality)
            loss = F.cross_entropy(logits, group_targets)
        loss.backward()
        grads = []
        for param in model.parameters():
            if param.grad is not None:
                grads.append(param.grad.data.clone())
            else:
                grads.append(None)
        group_grads.append(grads)
        group_losses.append(loss.item())
    weights = solve_min_norm_problem(group_grads, group_losses)
    return group_grads, weights

def combined_gradient_update(model, optimizer, batch, class_groups, n_groups, device,
                             normalization_type='l2', class_weight=0.7, sample_weight=0.3, loss_fn=None):
    group_grads, group_losses = get_group_grads(model, batch, class_groups, device, loss_fn=loss_fn)
    class_weights = solve_min_norm_problem(group_grads, group_losses, normalization_type)
    sample_grads, sample_weights = update_model_with_sample_level_mgda(model, batch, n_groups, device, loss_fn=loss_fn)
    optimizer.zero_grad()
    if group_grads:
        for i, weight in enumerate(class_weights):
            if i >= len(group_grads) or group_grads[i] is None:
                continue
            for j, (p, g) in enumerate(zip(model.parameters(), group_grads[i])):
                if g is None:
                    continue
                g_norm = torch.norm(g)
                if g_norm > 1.0:
                    g = g / g_norm
                if p.grad is None:
                    p.grad = class_weight * weight * g
                else:
                    p.grad += class_weight * weight * g
    if sample_grads is not None and sample_weights is not None:
        for i, weight in enumerate(sample_weights):
            if i >= len(sample_grads) or sample_grads[i] is None:
                continue
            for j, (p, g) in enumerate(zip(model.parameters(), sample_grads[i])):
                if g is None:
                    continue
                g_norm = torch.norm(g)
                if g_norm > 1.0:
                    g = g / g_norm
                if p.grad is None:
                    p.grad = sample_weight * weight * g
                else:
                    p.grad += sample_weight * weight * g
    optimizer.step()
    return class_weights, sample_weights
# ==== END: Dynamic gradient optimization functions ====