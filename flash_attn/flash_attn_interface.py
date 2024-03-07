import torch
import torch.nn as nn
import torch.nn.functional as F

import flash_attn_cuda


def _get_block_size(device, head_dim, is_dropout):
    assert head_dim % 8 == 0 and head_dim <= 128
    return 256 if head_dim <= 64 else 128


def _flash_attn_forward(q, k, v, out, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                        dropout_p, softmax_scale, causal, return_softmax, num_splits=0,
                        generator=None):
    """
    num_splits: how much to parallelize over the seqlen_q dimension. num_splits=0 means
    it will be set by an internal heuristic. We're exposing num_splits mostly for benchmarking.
    Don't change it unless you know what you're doing.
    """
    softmax_lse, rng_state, *rest = flash_attn_cuda.fwd(
        q, k, v, out, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p,
        softmax_scale, False, causal, return_softmax, num_splits, generator
    )
    # if out.isnan().any() or softmax_lse.isnan().any():
    #     breakpoint()
    S_dmask = rest[0] if return_softmax else None
    return out, softmax_lse, rng_state, S_dmask


def _flash_attn_backward(dout, q, k, v, out, softmax_lse, dq, dk, dv, cu_seqlens_q, cu_seqlens_k,
                         max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, causal,
                         rng_state=None, num_splits=0, generator=None):
    """
    num_splits: whether to parallelize over the seqlen_k dimension (num_splits > 1) or
    not (num_splits = 1). num_splits=0 means it will be set by an internal heuristic.
    Any value above 1 will call the same kernel (i.e. num_splits=2 would call the same kernel
    as num_splits=3), so effectively the choices are 0, 1, and 2.
    This hyperparameter can be tuned for performance, but default value (heuristic) should work fine.
    """
    dout = dout.contiguous()  # CUDA code assumes that dout is contiguous
    _, _, _, softmax_d = flash_attn_cuda.bwd(
        dout, q, k, v, out, softmax_lse, dq, dk, dv, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, False, causal,
        num_splits, generator, rng_state)
    # if dk.isnan().any() or dk.isnan().any() or dv.isnan().any() or softmax_d.isnan().any():
    #     breakpoint()
    return dq, dk, dv, softmax_d


class FlashAttnQKVPackedFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, qkv, cu_seqlens, max_seqlen, dropout_p, softmax_scale, causal,
                return_softmax, deterministic):
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)
        out, softmax_lse, rng_state, S_dmask = _flash_attn_forward(
            qkv[:, 0], qkv[:, 1], qkv[:, 2], torch.empty_like(qkv[:, 0]), cu_seqlens, cu_seqlens,
            max_seqlen, max_seqlen, dropout_p, softmax_scale, causal=causal,
            return_softmax=return_softmax
        )
        ctx.save_for_backward(qkv, out, softmax_lse, cu_seqlens, rng_state)
        ctx.dropout_p = dropout_p
        ctx.max_seqlen = max_seqlen
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.deterministic = deterministic
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        qkv, out, softmax_lse, cu_seqlens, rng_state = ctx.saved_tensors
        dqkv = torch.empty_like(qkv)
        _flash_attn_backward(
            dout, qkv[:, 0], qkv[:, 1], qkv[:, 2], out, softmax_lse,
            dqkv[:, 0], dqkv[:, 1], dqkv[:, 2], cu_seqlens, cu_seqlens,
            ctx.max_seqlen, ctx.max_seqlen, ctx.dropout_p, ctx.softmax_scale, ctx.causal,
            rng_state=rng_state, num_splits=1 if ctx.deterministic else 0,
        )
        return dqkv, None, None, None, None, None, None, None


class FlashAttnKVPackedFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, kv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p,
                softmax_scale, causal, return_softmax, deterministic):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        out, softmax_lse, rng_state, S_dmask = _flash_attn_forward(
            q, kv[:, 0], kv[:, 1], torch.empty_like(q), cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
            max_seqlen_k, dropout_p, softmax_scale, causal=causal, return_softmax=return_softmax
        )
        ctx.save_for_backward(q, kv, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state)
        ctx.dropout_p = dropout_p
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.deterministic = deterministic
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        q, kv, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state = ctx.saved_tensors
        dq = torch.empty_like(q)
        dkv = torch.empty_like(kv)
        _flash_attn_backward(
            dout, q, kv[:, 0], kv[:, 1], out, softmax_lse,
            dq, dkv[:, 0], dkv[:, 1], cu_seqlens_q, cu_seqlens_k,
            ctx.max_seqlen_q, ctx.max_seqlen_k, ctx.dropout_p, ctx.softmax_scale, ctx.causal,
            rng_state=rng_state, num_splits=1 if ctx.deterministic else 0,
        )
        return dq, dkv, None, None, None, None, None, None, None, None, None


class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, dropout_p, softmax_scale, causal, window_size, return_softmax):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_forward(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size=window_size,
            return_softmax=return_softmax and dropout_p > 0,
        )
        ctx.save_for_backward(q, k, v, out_padded, softmax_lse, rng_state)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, rng_state = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size,
            rng_state=rng_state,
        )
        dq = dq[..., : dout.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : dout.shape[-1]]
        dv = dv[..., : dout.shape[-1]]
        return dq, dk, dv, None, None, None, None, None, None, None, None, None



class FlashAttnQKVPackedSplitFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, qkv, cu_seqlens, max_seqlen0, max_seqlen1, batch_size0, dropout_p,
                softmax_scale, causal, return_softmax, deterministic):
        # Save rng_state because the backward pass will regenerate the dropout mask
        if dropout_p > 0:
            rng_state0 = torch.cuda.get_rng_state()
            generator1 = torch.Generator(device='cuda')
            rng_state1 = generator1.get_state()
        else:
            rng_state0, generator1, rng_state1 = None, None, None
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)
        out = torch.empty_like(qkv[:, 0])
        _, softmax_lse0, S_dmask0 = _flash_attn_forward(
            qkv[:, 0], qkv[:, 1], qkv[:, 2], out, cu_seqlens[:batch_size0 + 1],
            cu_seqlens[:batch_size0 + 1], max_seqlen0, max_seqlen0, dropout_p, softmax_scale,
            causal=causal, return_softmax=return_softmax
        )
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            _, softmax_lse1, S_dmask1 = _flash_attn_forward(
                qkv[:, 0], qkv[:, 1], qkv[:, 2], out, cu_seqlens[batch_size0:],
                cu_seqlens[batch_size0:], max_seqlen1, max_seqlen1, dropout_p, softmax_scale,
                causal=causal, return_softmax=return_softmax, generator=generator1
            )
        torch.cuda.current_stream().wait_stream(s)
        ctx.save_for_backward(qkv, out, softmax_lse0, softmax_lse1, cu_seqlens,
                              rng_state0, rng_state1)
        ctx.dropout_p = dropout_p
        ctx.max_seqlen0 = max_seqlen0
        ctx.max_seqlen1 = max_seqlen1
        ctx.batch_size0 = batch_size0
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.deterministic = deterministic
        if not return_softmax:
            return out
        else:
            max_seqlen_q = max(softmax_lse0.shape[2], softmax_lse1.shape[2])
            max_seqlen_k = max(S_dmask0.shape[3], S_dmask1.shape[3])
            softmax_lse = torch.cat([F.pad(softmax_lse0, (0, max_seqlen_q - softmax_lse0.shape[2])),
                                     F.pad(softmax_lse1, (0, max_seqlen_q - softmax_lse1.shape[2]))],
                                    dim=0)
            return out, softmax_lse, S_dmask0, S_dmask1

    @staticmethod
    def backward(ctx, dout, *args):
        qkv, out, softmax_lse0, softmax_lse1, cu_seqlens, rng_state0, rng_state1 = ctx.saved_tensors
        batch_size0 = ctx.batch_size0
        if rng_state0 is not None:
            cur_rng_state = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(rng_state0)
        if rng_state1 is not None:
            generator1 = torch.Generator(device='cuda')
            generator1.set_state(rng_state1)
        else:
            generator1 = None
        dqkv = torch.empty_like(qkv)
        _flash_attn_backward(
            dout, qkv[:, 0], qkv[:, 1], qkv[:, 2], out, softmax_lse0,
            dqkv[:, 0], dqkv[:, 1], dqkv[:, 2], cu_seqlens[:batch_size0 + 1],
            cu_seqlens[:batch_size0 + 1], ctx.max_seqlen0, ctx.max_seqlen0, ctx.dropout_p,
            ctx.softmax_scale, ctx.causal, num_splits=1 if ctx.deterministic else 0,
        )
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            _flash_attn_backward(
                dout, qkv[:, 0], qkv[:, 1], qkv[:, 2], out, softmax_lse1,
                dqkv[:, 0], dqkv[:, 1], dqkv[:, 2], cu_seqlens[batch_size0:],
                cu_seqlens[batch_size0:], ctx.max_seqlen1, ctx.max_seqlen1, ctx.dropout_p,
                ctx.softmax_scale, ctx.causal, generator=generator1,
                num_splits=1 if ctx.deterministic else 0,
            )
        torch.cuda.current_stream().wait_stream(s)
        if rng_state0 is not None:
            torch.cuda.set_rng_state(cur_rng_state)
        return dqkv, None, None, None, None, None, None, None, None, None


def flash_attn_unpadded_qkvpacked_func(qkv, cu_seqlens, max_seqlen, dropout_p, softmax_scale=None,
                                       causal=False, return_attn_probs=False, deterministic=False):
    """dropout_p should be set to 0.0 during evaluation
    Arguments:
        qkv: (total, 3, nheads, headdim), where total = total number of tokens in the batch.
        cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into qkv.
        max_seqlen: int. Maximum sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
        deterministic: bool. Whether or not to ensure deterministic execution.
    Return:
        out: (total, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return FlashAttnQKVPackedFunc.apply(qkv, cu_seqlens, max_seqlen, dropout_p, softmax_scale,
                                        causal, return_attn_probs, deterministic)


def flash_attn_unpadded_kvpacked_func(q, kv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                                      dropout_p, softmax_scale=None, causal=False,
                                      return_attn_probs=False, deterministic=False):
    """dropout_p should be set to 0.0 during evaluation
    Arguments:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        kv: (total_k, 2, nheads, headdim), where total_k = total number of key tokens in the batch.
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
        deterministic: bool. Whether or not to ensure deterministic execution.
    Return:
        out: (total_q, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return FlashAttnKVPackedFunc.apply(q, kv, cu_seqlens_q, cu_seqlens_k,
                                       max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, causal,
                                       return_attn_probs, deterministic)


def flash_attn_unpadded_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                             dropout_p, softmax_scale=None, causal=False, return_attn_probs=False,
                             deterministic=False):
    """dropout_p should be set to 0.0 during evaluation
    Arguments:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads, headdim), where total_k = total number of key tokens in the batch.
        v: (total_k, nheads, headdim), where total_k = total number of key tokens in the batch.
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
        deterministic: bool. Whether or not to ensure deterministic execution.
    Return:
        out: (total_q, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return FlashAttnFunc.apply(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                               dropout_p, softmax_scale, causal, return_attn_probs, deterministic)


def flash_attn_unpadded_qkvpacked_split_func(
        qkv, cu_seqlens, max_seqlen0, max_seqlen1, batch_size0, dropout_p, softmax_scale=None,
        causal=False, return_attn_probs=False, deterministic=False):
    """
    Split attention into 2 kernels running on 2 separate streams for performance reason:
    e.g., if the batch has some sequences of length <= 128 and some > 128, it might be faster to
    have one kernel dealing with seqlen <= 128 and one kernel for seqlen > 128.

    dropout_p should be set to 0.0 during evaluation.

    Arguments:
        qkv: (total, 3, nheads, headdim), where total = total number of tokens in the batch.
        cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into qkv.
        max_seqlen0: int. Maximum sequence length in 1st part of the batch.
        max_seqlen1: int. Maximum sequence length in 2nd part of the batch.
        batch_size0: int. Number of sequences in the 1st part of the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
        deterministic: bool. Whether or not to ensure deterministic execution.
    Return:
        out: (total, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return FlashAttnQKVPackedSplitFunc.apply(qkv, cu_seqlens, max_seqlen0, max_seqlen1, batch_size0,
                                             dropout_p, softmax_scale, causal, return_attn_probs,
                                             deterministic)


def flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    return_attn_probs=False,
):
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k: (batch_size, seqlen, nheads_k, headdim)
        v: (batch_size, seqlen, nheads_k, headdim)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return FlashAttnFunc.apply(
        q, k, v, dropout_p, softmax_scale, causal, window_size, return_attn_probs
    )
