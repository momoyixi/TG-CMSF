"""
here is the mian backbone for TGCMF
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...subNets import BertTextEncoder
from ...subNets.transformers_encoder.transformer import TransformerEncoder
#reverselayerF
class ReverseLayerF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

# attention control
class AttentionGate(nn.Module):
    def __init__(self, d_model: int, nhead: int = 4, pool_k: int = 5):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead)
        self.pool_k = pool_k
        self.linear_gate = nn.Linear(d_model, 1)

    def _sum_pool(self, x: torch.Tensor, k: int) -> torch.Tensor:
        # x: [S, N, C] -> SumPool along S with window k (same padding, stride=1)
        if k <= 1:
            return x
        S, N, C = x.shape
        w = torch.ones(C, 1, k, device=x.device, dtype=x.dtype)  # [C,1,k]
        x_ = x.permute(1, 2, 0)  # [N,C,S]
        pad = (k - 1) // 2
        y = torch.conv1d(x_, w, padding=pad, groups=C)  # [N,C,S]
        return y.permute(2, 0, 1)

    def _l2norm(self, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    def forward(self, It: torch.Tensor, Im: torch.Tensor) -> torch.Tensor:
        """
        It: [S_t, N, D], Im: [S_m, N, D]  -> gate_seq: [S_t, N, 1]
        """
        # 1. Use attention to calculate attention weights between It (text) and Im (audio/video)
        attn_output, attn_weights = self.attn(It, Im, Im)  # Attention between text and other modalities
        gate_seq = self.linear_gate(attn_output)  # [S_t, N, 1]
        
        # 2. Apply sum pooling to get the final gate value (for each sample)
        gate_seq = self._sum_pool(gate_seq, self.pool_k)  # SumPool with window size pool_k
        gate_seq = self._l2norm(gate_seq)  # L2 normalization
        gate_seq = torch.sigmoid(gate_seq)  # Sigmoid activation to get a gate value between 0 and 1

        return gate_seq


class TGCMF(nn.Module):
    def __init__(self, args):
        super(TGCMF, self).__init__()
        if args.use_bert:
            self.text_model = BertTextEncoder(use_finetune=args.use_finetune, transformers=args.transformers,
                                              pretrained=args.pretrained)
        self.use_bert = args.use_bert
        dst_feature_dims, nheads = args.dst_feature_dim_nheads
        if args.dataset_name == 'mosi':
            if args.need_data_aligned:
                self.len_l, self.len_v, self.len_a = 50, 50, 50
            else:
                self.len_l, self.len_v, self.len_a = 50, 500, 375
        if args.dataset_name == 'mosei':
            if args.need_data_aligned:
                self.len_l, self.len_v, self.len_a = 50, 50, 50
            else:
                self.len_l, self.len_v, self.len_a = 50, 500, 500
        self.orig_d_l, self.orig_d_a, self.orig_d_v = args.feature_dims
        self.d_l = self.d_a = self.d_v = dst_feature_dims
        self.num_heads = nheads
        self.layers = args.nlevels
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_a = args.attn_dropout_a
        self.attn_dropout_v = args.attn_dropout_v
        self.relu_dropout = args.relu_dropout
        self.embed_dropout = args.embed_dropout
        self.res_dropout = args.res_dropout
        self.output_dropout = args.output_dropout
        self.text_dropout = args.text_dropout
        self.attn_mask = args.attn_mask
        self.reverse_grad_weight = args.reverse_grad_weight
        self.use_cmd_sim=args.use_cmd_sim
        combined_dim_low = self.d_a
        combined_dim_high = self.d_a

        combined_dim = (self.d_l + self.d_a + self.d_v)

        #mask
        self.mask_enable     = True
        # mosi
        # self.mask_ratio      = 0.4
        #mosei
        self.mask_ratio      = 0.4
        self.mask_min_keep   = 1
        self.mask_gate_alpha = 0.3

        output_dim = 1

        # 1. Temporal convolutional layers for initial feature
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=args.conv1d_kernel_size_l, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=args.conv1d_kernel_size_a, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=args.conv1d_kernel_size_v, padding=0, bias=False)

        # 2. Modality-specific encoder
        self.encoder_s_l = self.get_network(self_type='l', layers=self.layers)
        self.encoder_s_v = self.get_network(self_type='v', layers=self.layers)
        self.encoder_s_a = self.get_network(self_type='a', layers=self.layers)

        #   Modality-shared encoder
        self.encoder_c = self.get_network(self_type='l', layers=self.layers)

        # tiny trick for modality embedding
        self.mod_type_embed = nn.ParameterDict({
            'l': nn.Parameter(torch.zeros(1,1,self.d_l)),
            'a': nn.Parameter(torch.zeros(1,1,self.d_a)),
            'v': nn.Parameter(torch.zeros(1,1,self.d_v)),
        })
        # mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.d_l))
        # Crossmodal generator
        self.gen_V_from_AL = CrossSharedWithL(d_model=self.d_l, nhead=nheads)
        self.gen_A_from_VL = CrossSharedWithL(d_model=self.d_l, nhead=nheads)

        # 3. Decoder for reconstruct three modalities
        self.decoder_l = nn.Conv1d(self.d_l * 2, self.d_l, kernel_size=1, padding=0, bias=False)
        self.decoder_v = nn.Conv1d(self.d_v * 2, self.d_v, kernel_size=1, padding=0, bias=False)
        self.decoder_a = nn.Conv1d(self.d_a * 2, self.d_a, kernel_size=1, padding=0, bias=False)

        # for calculate cosine sim between s_x
        self.proj_cosine_l = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), combined_dim_low)
        self.proj_cosine_v = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), combined_dim_low)
        self.proj_cosine_a = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), combined_dim_low)

        # for align c_l, c_v, c_a
        self.align_c_l = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), combined_dim_low)
        self.align_c_v = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), combined_dim_low)
        self.align_c_a = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), combined_dim_low)

        self.self_attentions_c_l = self.get_network(self_type='l')
        self.self_attentions_c_v = self.get_network(self_type='v')
        self.self_attentions_c_a = self.get_network(self_type='a')

        self.proj1_c = nn.Linear(self.d_l * 3, self.d_l * 3)
        self.proj2_c = nn.Linear(self.d_l * 3, self.d_l * 3)
        self.out_layer_c = nn.Linear(self.d_l * 3, output_dim)
        # prompt lite
        self.prompt_lite = PromptLite(d_model=self.d_l, P=8)  # P 4/8/12
        #discriminator
        self.domain_discriminator = nn.Sequential(
            nn.Linear(self.d_l, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # 4 Multimodal Crossmodal Attentions
        self.trans_l_with_a = self.get_network(self_type='la', layers=self.layers)
        self.trans_l_with_v = self.get_network(self_type='lv', layers=self.layers)
        self.trans_a_with_l = self.get_network(self_type='al')
        self.trans_a_with_v = self.get_network(self_type='av')
        self.trans_v_with_l = self.get_network(self_type='vl')
        self.trans_v_with_a = self.get_network(self_type='va')
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=self.layers)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)

        # 5. fc layers for shared features
        self.proj1_l_low = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), combined_dim_low)
        self.proj2_l_low = nn.Linear(combined_dim_low, combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1))
        self.out_layer_l_low = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), output_dim)
        self.proj1_v_low = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), combined_dim_low)
        self.proj2_v_low = nn.Linear(combined_dim_low, combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1))
        self.out_layer_v_low = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), output_dim)
        self.proj1_a_low = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), combined_dim_low)
        self.proj2_a_low = nn.Linear(combined_dim_low, combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1))
        self.out_layer_a_low = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), output_dim)

        # 6. fc layers for specific features
        self.proj1_l_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_l_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_l_high = nn.Linear(combined_dim_high, output_dim)
        self.proj1_v_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_v_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_v_high = nn.Linear(combined_dim_high, output_dim)
        self.proj1_a_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_a_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_a_high = nn.Linear(combined_dim_high, output_dim)

        # 7. project for fusion (special heads)
        self.projector_l = nn.Linear(self.d_l, self.d_l)
        self.projector_v = nn.Linear(self.d_v, self.d_v)
        self.projector_a = nn.Linear(self.d_a, self.d_a)
        self.projector_c = nn.Linear(3 * self.d_l, 3 * self.d_l)

        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

        self.attention_gate_a_final = AttentionGate(d_model=self.d_l, nhead=nheads, pool_k=5)
        self.attention_gate_v_final = AttentionGate(d_model=self.d_l, nhead=nheads, pool_k=5)

    def _make_balanced_complementary_masks(self, S: int, device, ratio: float = 0.22):
        num = max(1, int(round(ratio * S)))
        num = min(num, S // 2)
        perm = torch.randperm(S, device=device)
        a_mask_idx = perm[:num]
        v_mask_idx = perm[num:2*num]

        mask_a_keep = torch.ones(S, dtype=torch.bool, device=device)
        mask_v_keep = torch.ones(S, dtype=torch.bool, device=device)
        mask_a_keep[a_mask_idx] = False
        mask_v_keep[v_mask_idx] = False
        return mask_a_keep, mask_v_keep

    def _apply_mask_token(self, c: torch.Tensor, keep_mask_1d: torch.Tensor):
        S, N, D = c.shape
        keep_ = keep_mask_1d.view(S, 1, 1)
        masked_pos = ~keep_
        c_masked = c.clone()
        mask_tok = self.mask_token.expand(S, N, D) if hasattr(self, "mask_token") else 0.0
        c_masked[masked_pos.expand_as(c_masked)] = mask_tok[masked_pos.expand_as(c_masked)]
        return c_masked, masked_pos

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def forward(self, text, audio, video):
        # extraction
        if self.use_bert:
            text = self.text_model(text)
        x_l = F.dropout(text.transpose(1, 2), p=self.text_dropout, training=self.training)
        x_a = audio.transpose(1, 2)
        x_v = video.transpose(1, 2)

        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)

        proj_x_l = proj_x_l.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_a = proj_x_a.permute(2, 0, 1)

        # disentanglement
        s_l = self.encoder_s_l(proj_x_l)
        s_v = self.encoder_s_v(proj_x_v)
        s_a = self.encoder_s_a(proj_x_a)

        inp_l = proj_x_l + self.mod_type_embed['l']
        inp_v = proj_x_v + self.mod_type_embed['v']
        inp_a = proj_x_a + self.mod_type_embed['a']
        # shared encoder
        c_l = self.encoder_c(inp_l)
        c_v = self.encoder_c(inp_v)
        c_a = self.encoder_c(inp_a)

        s_l = s_l.permute(1, 2, 0)
        s_v = s_v.permute(1, 2, 0)
        s_a = s_a.permute(1, 2, 0)

        c_l = c_l.permute(1, 2, 0)
        c_v = c_v.permute(1, 2, 0)
        c_a = c_a.permute(1, 2, 0)
        c_list = [c_l, c_v, c_a]
        #discriminator
        if not self.use_cmd_sim:
            reversed_c_l = ReverseLayerF.apply(c_l, self.reverse_grad_weight)
            reversed_c_v = ReverseLayerF.apply(c_v, self.reverse_grad_weight)
            reversed_c_a = ReverseLayerF.apply(c_a, self.reverse_grad_weight)
            

            # 调整维度 (batch_size, feature_dim, seq_len) -> (batch_size, feature_dim)
            domain_input_l = reversed_c_l.mean(dim=-1)
            domain_input_v = reversed_c_v.mean(dim=-1)
            domain_input_a = reversed_c_a.mean(dim=-1)
            
            self.domain_label_l = self.domain_discriminator(domain_input_l)
            self.domain_label_v = self.domain_discriminator(domain_input_v)
            self.domain_label_a = self.domain_discriminator(domain_input_a)
        else:
            self.domain_label_l = None
            self.domain_label_v = None
            self.domain_label_a = None

        c_l_sim = self.align_c_l(c_l.contiguous().view(x_l.size(0), -1))
        c_v_sim = self.align_c_v(c_v.contiguous().view(x_l.size(0), -1))
        c_a_sim = self.align_c_a(c_a.contiguous().view(x_l.size(0), -1))

        recon_l = self.decoder_l(torch.cat([s_l, c_list[0]], dim=1))
        recon_v = self.decoder_v(torch.cat([s_v, c_list[1]], dim=1))
        recon_a = self.decoder_a(torch.cat([s_a, c_list[2]], dim=1))

        recon_l = recon_l.permute(2, 0, 1)
        recon_v = recon_v.permute(2, 0, 1)
        recon_a = recon_a.permute(2, 0, 1)

        s_l_r = self.encoder_s_l(recon_l).permute(1, 2, 0)
        s_v_r = self.encoder_s_v(recon_v).permute(1, 2, 0)
        s_a_r = self.encoder_s_a(recon_a).permute(1, 2, 0)

        s_l = s_l.permute(2, 0, 1)
        s_v = s_v.permute(2, 0, 1)
        s_a = s_a.permute(2, 0, 1)

        c_l = c_l.permute(2, 0, 1)
        c_v = c_v.permute(2, 0, 1)
        c_a = c_a.permute(2, 0, 1)

        S, N, D = c_a.shape
        dev = c_a.device
        prompt_kv = self.prompt_lite(c_l)  # [P,N,D]

        a_mask_pos = torch.zeros(S, 1, 1, dtype=torch.bool, device=dev)
        v_mask_pos = torch.zeros(S, 1, 1, dtype=torch.bool, device=dev)
        pred_a = c_a
        pred_v = c_v
        c_a_used = c_a
        c_v_used = c_v

        if self.training and getattr(self, "mask_enable", False):
            mask_a_keep, mask_v_keep = self._make_balanced_complementary_masks(S, dev, ratio=self.mask_ratio)
            c_a_masked, a_mask_pos = self._apply_mask_token(c_a, mask_a_keep)
            c_v_masked, v_mask_pos = self._apply_mask_token(c_v, mask_v_keep)

            pred_v = self.gen_V_from_AL(c_l, c_a_masked, cond_keep_1d=mask_a_keep, prompt_kv=prompt_kv)
            pred_a = self.gen_A_from_VL(c_l, c_v_masked, cond_keep_1d=mask_v_keep, prompt_kv=prompt_kv)

            alpha = self.mask_gate_alpha
            idx_v = v_mask_pos.expand_as(c_v)
            idx_a = a_mask_pos.expand_as(c_a)
            c_v_used = torch.where(idx_v, alpha*pred_v + (1-alpha)*c_v, c_v)
            c_a_used = torch.where(idx_a, alpha*pred_a + (1-alpha)*c_a, c_a)

        hs_l_low = c_l.transpose(0, 1).contiguous().view(x_l.size(0), -1)
        repr_l_low = self.proj1_l_low(hs_l_low)
        hs_proj_l_low = self.proj2_l_low(
            F.dropout(F.relu(repr_l_low, inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_l_low += hs_l_low
        logits_l_low = self.out_layer_l_low(hs_proj_l_low)

        hs_v_low = c_v.transpose(0, 1).contiguous().view(x_v.size(0), -1)
        repr_v_low = self.proj1_v_low(hs_v_low)
        hs_proj_v_low = self.proj2_v_low(
            F.dropout(F.relu(repr_v_low, inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_v_low += hs_v_low
        logits_v_low = self.out_layer_v_low(hs_proj_v_low)

        hs_a_low = c_a.transpose(0, 1).contiguous().view(x_a.size(0), -1)
        repr_a_low = self.proj1_a_low(hs_a_low)
        hs_proj_a_low = self.proj2_a_low(
            F.dropout(F.relu(repr_a_low, inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_a_low += hs_a_low
        logits_a_low = self.out_layer_a_low(hs_proj_a_low)

        c_l_att = self.self_attentions_c_l(c_l)
        if type(c_l_att) == tuple:
            c_l_att = c_l_att[0]
        c_l_att = c_l_att[-1]

        c_v_att = self.self_attentions_c_v(c_v)
        if type(c_v_att) == tuple:
            c_v_att = c_v_att[0]
        c_v_att = c_v_att[-1]

        c_a_att = self.self_attentions_c_a(c_a)
        if type(c_a_att) == tuple:
            c_a_att = c_a_att[0]
        c_a_att = c_a_att[-1]

        c_fusion = torch.cat([c_l_att, c_v_att, c_a_att], dim=1)  
        c_proj = self.proj2_c(
            F.dropout(F.relu(self.proj1_c(c_fusion), inplace=True), p=self.output_dropout, training=self.training))
        c_proj += c_fusion
        logits_c = self.out_layer_c(c_proj)


        h_ls = s_l
        h_ls = self.trans_l_mem(h_ls)
        if type(h_ls) == tuple:
            h_ls = h_ls[0]
        last_h_l = last_hs = h_ls[-1]

        h_l_with_as = self.trans_l_with_a(s_l, s_a, s_a)
        h_as = h_l_with_as
        h_as = self.trans_a_mem(h_as)
        if type(h_as) == tuple:
            h_as = h_as[0]
        last_h_a = last_hs = h_as[-1]

        h_l_with_vs = self.trans_l_with_v(s_l, s_v, s_v)
        h_vs = h_l_with_vs
        h_vs = self.trans_v_mem(h_vs)
        if type(h_vs) == tuple:
            h_vs = h_vs[0]
        last_h_v = last_hs = h_vs[-1]

        hs_proj_l_high = self.proj2_l_high(
            F.dropout(F.relu(self.proj1_l_high(last_h_l), inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_l_high += last_h_l
        logits_l_high = self.out_layer_l_high(hs_proj_l_high)

        hs_proj_v_high = self.proj2_v_high(
            F.dropout(F.relu(self.proj1_v_high(last_h_v), inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_v_high += last_h_v
        logits_v_high = self.out_layer_v_high(hs_proj_v_high)

        hs_proj_a_high = self.proj2_a_high(
            F.dropout(F.relu(self.proj1_a_high(last_h_a), inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_a_high += last_h_a
        logits_a_high = self.out_layer_a_high(hs_proj_a_high)

        gate_a_seq = self.attention_gate_a_final(c_l, c_a)   
        gate_v_seq = self.attention_gate_v_final(c_l, c_v)   
        gate_a = gate_a_seq.mean(dim=0)  
        gate_v = gate_v_seq.mean(dim=0)  
        last_h_l = torch.sigmoid(self.projector_l(hs_proj_l_high))
        last_h_v = torch.sigmoid(self.projector_v(hs_proj_v_high))
        last_h_a = torch.sigmoid(self.projector_a(hs_proj_a_high))

        last_h_a = last_h_a * gate_a     
        last_h_v = last_h_v * gate_v      

        last_hs = torch.cat([last_h_l, last_h_v, last_h_a], dim=1)

        # prediction
        last_hs_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_hs), inplace=True), p=self.output_dropout, training=self.training))
        last_hs_proj += last_hs
        output = self.out_layer(last_hs_proj)

        res = {
            'origin_l': proj_x_l,
            'origin_v': proj_x_v,
            'origin_a': proj_x_a,
            's_l': s_l,
            's_v': s_v,
            's_a': s_a,
            'c_l': c_l,
            'c_v': c_v,
            'c_a': c_a,
            'domain_label_l': self.domain_label_l,
            'domain_label_v': self.domain_label_v,
            'domain_label_a': self.domain_label_a,
            's_l_r': s_l_r,
            's_v_r': s_v_r,
            's_a_r': s_a_r,
            'recon_l': recon_l,
            'recon_v': recon_v,
            'recon_a': recon_a,
            'c_l_sim': c_l_sim,
            'c_v_sim': c_v_sim,
            'c_a_sim': c_a_sim,
            'logits_l_hetero': logits_l_high,
            'logits_v_hetero': logits_v_high,
            'logits_a_hetero': logits_a_high,
            'logits_c': logits_c,           
            'c_a_pred': pred_a,
            'c_v_pred': pred_v,
            'c_a_used': c_a_used,
            'c_v_used': c_v_used,
            'a_mask_pos': a_mask_pos,
            'v_mask_pos': v_mask_pos,
            'output_logit': output,
            'gate_a': gate_a,         
            'gate_v': gate_v,
            'gate_a_seq': gate_a_seq, 
            'gate_v_seq': gate_v_seq
        }
        return res


class CrossSharedWithL(nn.Module):
    def __init__(self, d_model, nhead=4, nlayers=1, ff_mult=4):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, ff_mult*d_model, batch_first=False)
        self.self_on_L = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.cross = nn.MultiheadAttention(d_model, nhead, batch_first=False)
        self.ffn = nn.Sequential(nn.Linear(d_model, ff_mult*d_model), nn.ReLU(True), nn.Linear(ff_mult*d_model, d_model))
        self.ln1 = nn.LayerNorm(d_model); self.ln2 = nn.LayerNorm(d_model); self.ln3 = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, c_l, c_cond, cond_keep_1d: torch.Tensor, prompt_kv: torch.Tensor = None):
        L = self.self_on_L(c_l)
        N = c_cond.size(1)
        if prompt_kv is not None:
            Kcat = torch.cat([prompt_kv, c_cond], dim=0)  # [P+S,N,D]
            pad_prompt = torch.zeros(N, prompt_kv.size(0), dtype=torch.bool, device=c_cond.device)
            pad_cond   = (~cond_keep_1d).unsqueeze(0).expand(N, -1)  # [N,S]
            key_pad    = torch.cat([pad_prompt, pad_cond], dim=1)    # [N,P+S]
        else:
            Kcat = c_cond
            key_pad = (~cond_keep_1d).unsqueeze(0).expand(N, -1)

        attn_out, _ = self.cross(query=L, key=Kcat, value=Kcat, key_padding_mask=key_pad)
        x = self.ln1(L + attn_out)
        x = self.ln2(x + self.ffn(x))
        return self.out(self.ln3(x))


class PromptLite(nn.Module):
    def __init__(self, d_model: int, P: int = 8, mode: str = "global"):
        super().__init__()
        self.mode, self.P, self.D = mode, P, d_model
        self.prompt = nn.Parameter(torch.randn(P, d_model) * 0.02)

    @torch.no_grad()
    def reset_parameters(self):
        nn.init.normal_(self.prompt, mean=0.0, std=0.02)

    def forward(self, c_l: torch.Tensor):
        N = c_l.size(1)
        return self.prompt.unsqueeze(1).expand(self.P, N, self.D)  # [P,N,D]
