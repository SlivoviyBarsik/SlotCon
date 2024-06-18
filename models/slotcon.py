import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torchvision

from nerv.models.transformer import build_transformer_encoder

class DINOHead(nn.Module):
    def __init__(self, in_dim, use_bn=True, nlayers=3, hidden_dim=4096, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        return x

class DINOHead2d(nn.Module):
    def __init__(self, in_dim, use_bn=True, nlayers=3, hidden_dim=4096, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Conv2d(in_dim, bottleneck_dim, 1)
        else:
            layers = [nn.Conv2d(in_dim, hidden_dim, 1)]
            if use_bn:
                layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Conv2d(hidden_dim, hidden_dim, 1))
                if use_bn:
                    layers.append(nn.BatchNorm2d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Conv2d(hidden_dim, bottleneck_dim, 1))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        return x

class SemanticGrouping(nn.Module):
    def __init__(self, num_slots, dim_slot, temp=0.07, eps=1e-6):
        super().__init__()
        self.num_slots = num_slots
        self.dim_slot = dim_slot
        self.temp = temp
        self.eps = eps

        self.slot_embed = nn.Embedding(num_slots, dim_slot)

    def forward(self, x):
        x_prev = x
        slots = self.slot_embed(torch.arange(0, self.num_slots, device=x.device)).unsqueeze(0).repeat(x.size(0), 1, 1)
        dots = torch.einsum('bkd,bdhw->bkhw', F.normalize(slots, dim=2), F.normalize(x, dim=1))
        attn = (dots / self.temp).softmax(dim=1) + self.eps
        slots = torch.einsum('bdhw,bkhw->bkd', x_prev, attn / attn.sum(dim=(2, 3), keepdim=True))
        return slots, dots

class SlotCon(nn.Module):
    def __init__(self, encoder, args):
        super().__init__()

        self.dim_hidden = args.dim_hidden
        self.dim_out = args.dim_out
        self.teacher_momentum = args.teacher_momentum

        if args.arch in ('resnet_small', 'resnet18', 'resnet34'):
            self.num_channels = 512  
        elif args.arch == 'spr_cnn':
            self.num_channels = 64
        elif args.arch == 'resnet3l':
            self.num_channels = 256 
        else:
            self.num_channels = 2048
            
        self.encoder_q = encoder(head_type='early_return')
        self.encoder_k = encoder(head_type='early_return')

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder_q)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder_k)

        self.group_loss_weight = args.group_loss_weight
        self.student_temp = args.student_temp
        self.teacher_temp = args.teacher_temp
            
        self.projector_q = DINOHead2d(self.num_channels, hidden_dim=self.dim_hidden, bottleneck_dim=self.dim_out)
        self.projector_k = DINOHead2d(self.num_channels, hidden_dim=self.dim_hidden, bottleneck_dim=self.dim_out)

        for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
            
        nn.SyncBatchNorm.convert_sync_batchnorm(self.projector_q)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.projector_k)

        self.num_prototypes = args.num_prototypes
        self.center_momentum = args.center_momentum
        self.register_buffer("center", torch.zeros(1, self.num_prototypes))
        self.grouping_q = SemanticGrouping(self.num_prototypes, self.dim_out, self.teacher_temp)
        self.grouping_k = SemanticGrouping(self.num_prototypes, self.dim_out, self.teacher_temp)
        self.predictor_slot = DINOHead(self.dim_out, hidden_dim=self.dim_hidden, bottleneck_dim=self.dim_out)

        nn.SyncBatchNorm.convert_sync_batchnorm(self.predictor_slot)
            
        for param_q, param_k in zip(self.grouping_q.parameters(), self.grouping_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.K = int(args.num_instances * 1. / args.world_size / args.batch_size * args.epochs)
        self.k = int(args.num_instances * 1. / args.world_size / args.batch_size * (args.start_epoch - 1))

    def re_init(self, args):
        self.k = int(args.num_instances * 1. / args.world_size / args.batch_size * (args.start_epoch - 1))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        momentum = 1. - (1. - self.teacher_momentum) * (math.cos(math.pi * self.k / self.K) + 1.) * 0.5
        self.k += 1
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)
        for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)
        for param_q, param_k in zip(self.grouping_q.parameters(), self.grouping_k.parameters()):
            param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)  

    def invaug(self, x, coords, flags):
        N, C, H, W = x.shape

        batch_idxs = torch.arange(N, device=coords.device).view(N, 1)
        coords_rescaled = coords.clone()
        coords_rescaled[:, 0] = coords_rescaled[:, 0] * W  # x1
        coords_rescaled[:, 2] = coords_rescaled[:, 2] * W  # x2
        coords_rescaled[:, 1] = coords_rescaled[:, 1] * H  # y1
        coords_rescaled[:, 3] = coords_rescaled[:, 3] * H  # y2
        coords_with_idxs = torch.cat([batch_idxs, coords_rescaled], dim=1)
        
        x_aligned = torchvision.ops.roi_align(x, coords_with_idxs, (H, W), aligned=True)
        x_flipped = torch.stack([feat.flip(-1) if flag else feat for feat, flag in zip(x_aligned, flags)])
        return x_flipped

    def self_distill(self, q, k):
        q = F.log_softmax(q / self.student_temp, dim=-1)
        k = F.softmax((k - self.center) / self.teacher_temp, dim=-1)
        return torch.sum(-k * q, dim=-1).mean()

    def ctr_loss_filtered(self, q, k, score_q, score_k, tau=0.2):
        q = q.flatten(0, 1)
        k = F.normalize(k.flatten(0, 1), dim=1)

        mask_q = (torch.zeros_like(score_q).scatter_(1, score_q.argmax(1, keepdim=True), 1).sum(-1).sum(-1) > 0).long().detach()
        mask_k = (torch.zeros_like(score_k).scatter_(1, score_k.argmax(1, keepdim=True), 1).sum(-1).sum(-1) > 0).long().detach()
        mask_intersection = (mask_q * mask_k).view(-1)
        idxs_q = mask_intersection.nonzero().squeeze(-1)

        mask_k = concat_all_gather(mask_k.view(-1))
        idxs_k = mask_k.nonzero().squeeze(-1)

        N = k.shape[0]
        logits = torch.einsum('nc,mc->nm', [F.normalize(self.predictor_slot(q[idxs_q]), dim=1), concat_all_gather(k)[idxs_k]]) / tau
        labels = mask_k.cumsum(0)[idxs_q + N * torch.distributed.get_rank()] - 1
        return F.cross_entropy(logits, labels) * (2 * tau)

    def forward(self, input, action=None, return_q1_aligned=False, *args):
        crops, coords, flags = input
        if len(crops[0].shape) > 4: 
            crops = [crops[0][:,0], crops[1][:,0]]  # ignore stack
        x1, x2 = self.projector_q(self.encoder_q(crops[0])), self.projector_q(self.encoder_q(crops[1]))
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            y1, y2 = self.projector_k(self.encoder_k(crops[0])), self.projector_k(self.encoder_k(crops[1]))
            
        (q1, score_q1), (q2, score_q2) = self.grouping_q(x1), self.grouping_q(x2)
        q1_aligned, q2_aligned = self.invaug(score_q1, coords[0], flags[0]), self.invaug(score_q2, coords[1], flags[1])
        with torch.no_grad():
            (k1, score_k1), (k2, score_k2) = self.grouping_k(y1), self.grouping_k(y2)
            k1_aligned, k2_aligned = self.invaug(score_k1, coords[0], flags[0]), self.invaug(score_k2, coords[1], flags[1])
        
        loss = self.group_loss_weight * self.self_distill(q1_aligned.permute(0, 2, 3, 1).flatten(0, 2), k2_aligned.permute(0, 2, 3, 1).flatten(0, 2)) \
             + self.group_loss_weight * self.self_distill(q2_aligned.permute(0, 2, 3, 1).flatten(0, 2), k1_aligned.permute(0, 2, 3, 1).flatten(0, 2))

        self.update_center(torch.cat([score_k1, score_k2]).permute(0, 2, 3, 1).flatten(0, 2))

        loss += (1. - self.group_loss_weight) * self.ctr_loss_filtered(q1, k2, score_q1, score_k2) \
              + (1. - self.group_loss_weight) * self.ctr_loss_filtered(q2, k1, score_q2, score_k1)
        
        if return_q1_aligned:
            return loss, x1, q1, q1_aligned
        return loss
    
    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

class SlotConEval(nn.Module):
    def __init__(self, encoder, args):
        super().__init__()

        self.dim_hidden = args.dim_hidden
        self.dim_out = args.dim_out

        if args.arch in ('resnet_small', 'resnet18', 'resnet34'):
            self.num_channels = 512  
        elif args.arch == 'spr_cnn':
            self.num_channels = 64
        elif args.arch == 'resnet3l':
            self.num_channels = 256 
        else:
            self.num_channels = 2048

        self.encoder_k = encoder(head_type='early_return')
        for param_k in self.encoder_k.parameters():
            param_k.requires_grad = False  # not update by gradient

        self.projector_k = DINOHead2d(self.num_channels, hidden_dim=self.dim_hidden, bottleneck_dim=self.dim_out)
        for param_k in self.projector_k.parameters():
            param_k.requires_grad = False  # not update by gradient

        self.num_prototypes = args.num_prototypes
        self.grouping_k = SemanticGrouping(self.num_prototypes, self.dim_out)
        for param_k in self.grouping_k.parameters():
            param_k.requires_grad = False  # not update by gradient

    def forward(self, x):
        with torch.no_grad():
            slots, probs = self.grouping_k(self.projector_k(self.encoder_k(x)))
            return probs
        

class SlotConSPR(SlotCon):
    def __init__(self, encoder, args):
        super().__init__(encoder, args)

        self.transformer_transition = build_transformer_encoder(
            input_len=args.num_prototypes, pos_enc=None, d_model=args.dim_out+2, 
            ffn_dim=args.dim_out+2, num_layers=args.transition_enc_layers, 
            norm_first=True, norm_last=False, num_heads=args.transition_enc_heads
        )

        self.action_emb = nn.Embedding(18, args.dim_out+2)

        self.grid = None 

        self.branch_lambda = args.branch_lambda
        self.spr_lambda = args.spr_lambda

    def forward(self, input, action):
        slotcon_loss, t1, t1_slots, t1_scores_aligned = super().forward(input, return_q1_aligned=True)
        crops, coords, flags = input

        act_slots = self.action_emb(action.cuda())  # [N, 1, D+2]
        
        if self.grid is None:
            self.grid = torch.linspace(0, 1, t1.shape[-1], device=t1.device)  # [H]
            self.grid = self.grid.unsqueeze(0).unsqueeze(0)  # [1, 1, H]

        with torch.no_grad():  # no gradient to keys
            t1_k, t2_k = self.projector_q(self.encoder_q(crops[0][:,-1])), self.projector_k(self.encoder_k(crops[1][:,-1]))
            (t1_k_slots, t1_k_scores), (t2_k_slots, t2_k_scores) = self.grouping_q(t1_k), self.grouping_k(t2_k)
        
        t1_masks = torch.zeros_like(t1_scores_aligned).scatter_(1, t1_scores_aligned.argmax(1, keepdim=True), 1).detach()  # [N, K, H, W]
        t1_slot_masks = (t1_masks.sum(-1).sum(-1) == 0)  # [N, K]
        
        x_pos = ((t1_masks * self.grid.unsqueeze(-1)).sum(dim=[2,3]) / (t1_masks.sum(dim=[2,3]) + 1e-6)).unsqueeze(-1)  # [N, K, 1]
        y_pos = ((t1_masks * self.grid.unsqueeze(-2)).sum(dim=[2,3]) / (t1_masks.sum(dim=[2,3]) + 1e-6)).unsqueeze(-1)  # [N, K, 1]

        t1_slots_wpos = torch.concat([t1_slots, x_pos, y_pos], -1)  # [N, K, D+2]
        tot_slots = torch.concat([t1_slots_wpos, act_slots], 1)  # [N, K+1, D+2]
        tot_slots_pred = self.transformer_transition(tot_slots, src_key_padding_mask=torch.concat([t1_slot_masks, torch.zeros(tot_slots.shape[0], 1, device='cuda')], -1))  # [N, K+1, D+2]
        t1_k_predicted = tot_slots_pred[:,:-1]  # [N, K, D+2]

        t1_k_scores_aligned, t2_k_scores_aligned = self.invaug(t1_k_scores, coords[0], flags[0]), self.invaug(t2_k_scores, coords[1], flags[1])
        t1_k_masks = torch.zeros_like(t1_k_scores_aligned).scatter_(1, t1_k_scores_aligned.argmax(1, keepdim=True), 1).detach()  # [N, K]
        t2_k_masks = torch.zeros_like(t2_k_scores_aligned).scatter_(1, t2_k_scores_aligned.argmax(1, keepdim=True), 1).detach()  # [N, K]

        x1_pos = ((t1_k_masks * self.grid.unsqueeze(-1)).sum(dim=[2,3]) / (t1_k_masks.sum(dim=[2,3]) + 1e-6)).unsqueeze(-1)  # [N, K, 1]
        y1_pos = ((t1_k_masks * self.grid.unsqueeze(-2)).sum(dim=[2,3]) / (t1_k_masks.sum(dim=[2,3]) + 1e-6)).unsqueeze(-1)  # [N, K, 1]

        x2_pos = ((t2_k_masks * self.grid.unsqueeze(-1)).sum(dim=[2,3]) / (t2_k_masks.sum(dim=[2,3]) + 1e-6)).unsqueeze(-1)  # [N, K, 1]
        y2_pos = ((t2_k_masks * self.grid.unsqueeze(-2)).sum(dim=[2,3]) / (t2_k_masks.sum(dim=[2,3]) + 1e-6)).unsqueeze(-1)  # [N, K, 1]

        t1_k_wpos = torch.concat([t1_k_slots, x1_pos, y1_pos], -1)
        t2_k_wpos = torch.concat([t2_k_slots, x2_pos, y2_pos], -1)

        t1_k_wpos = F.normalize(t1_k_wpos.float(), p=2., dim=-1, eps=1e-3)
        t2_k_wpos = F.normalize(t2_k_wpos.float(), p=2., dim=-1, eps=1e-3)
        t1_k_predicted = F.normalize(t1_k_predicted.float(), p=2., dim=-1, eps=1e-3)

        spr_loss_1 = F.mse_loss(t1_k_predicted, t1_k_wpos * (t1_k_masks.sum(-1).sum(-1) > 0).unsqueeze(-1).float())
        spr_loss_2 = F.mse_loss(t1_k_predicted, t2_k_wpos * (t2_k_masks.sum(-1).sum(-1) > 0).unsqueeze(-1).float())

        spr_loss = self.branch_lambda * (spr_loss_1) + (1. - self.branch_lambda) * spr_loss_2

        return self.spr_lambda * spr_loss + (1. - self.spr_lambda) * slotcon_loss