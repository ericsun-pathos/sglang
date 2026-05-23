from typing import Optional

import torch
import torch.nn.functional as F

from sglang.srt.utils import is_cuda
from sglang.srt.utils.custom_op import register_custom_op

_is_cuda = is_cuda()

if _is_cuda:
    from sgl_kernel import moe_sum_reduce

    from sglang.jit_kernel.activation import silu_and_mul
    from sglang.jit_kernel.moe_wna16_marlin import moe_wna16_marlin_gemm


def get_scalar_type(num_bits: int, has_zp: bool, scales: Optional[torch.Tensor] = None):
    from sgl_kernel.scalar_type import scalar_types

    if (
        not has_zp
        and num_bits == 4
        and scales is not None
        and scales.dtype == torch.float8_e8m0fnu
    ):
        return scalar_types.float4_e2m1f
    if has_zp:
        assert num_bits == 4
        return scalar_types.uint4
    else:
        return scalar_types.uint4b8 if num_bits == 4 else scalar_types.uint8b128


def swiglu_limit_func(
    output: torch.Tensor,
    input: torch.Tensor,  # first half is gate, second half is up
    swiglu_limit: float = 0.0,
) -> None:
    d = input.shape[1] // 2
    gate = input[:, :d]
    up = input[:, d:]

    if swiglu_limit > 0:
        gate = torch.clamp(gate, max=swiglu_limit)
        up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)

    output.copy_(F.silu(gate) * up)


@register_custom_op(out_shape="hidden_states")
def fused_marlin_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    gating_output: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    g_idx1: Optional[torch.Tensor] = None,
    g_idx2: Optional[torch.Tensor] = None,
    sort_indices1: Optional[torch.Tensor] = None,
    sort_indices2: Optional[torch.Tensor] = None,
    w1_zeros: Optional[torch.Tensor] = None,
    w2_zeros: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    num_bits: int = 8,
    is_k_full: bool = True,
    inplace: bool = False,
    routed_scaling_factor: Optional[float] = None,
    clamp_limit: Optional[float] = None,
) -> torch.Tensor:
    """
    This function computes a Mixture of Experts (MoE) layer using two sets of
    weights, w1 and w2, and top-k gating mechanism.

    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.
    - w1 (torch.Tensor): The first set of expert weights.
    - w2 (torch.Tensor): The second set of expert weights.
    - w1_scale (torch.Tensor): Scale to be used for w1.
    - w2_scale (torch.Tensor): Scale to be used for w2.
    - gating_output (torch.Tensor): The output of the gating operation
        (before softmax).
    - g_idx1 (Optional[torch.Tensor]): The first set of act_order indices.
    - g_idx2 (Optional[torch.Tensor]): The second set of act_order indices.
    - sort_indices1 (Optional[torch.Tensor]): The first act_order input
        permutation.
    - sort_indices2 (Optional[torch.Tensor]): The second act_order input
        permutation.
    - topk_weights (torch.Tensor): Top-k weights.
    - topk_ids (torch.Tensor): Indices of topk-k elements.
    - w1_zeros (Optional[torch.Tensor]): Optional zero points to be used for w1.
    - w2_zeros (Optional[torch.Tensor]): Optional zero points to be used for w2.
    - num_bits (int): The number of bits in expert weights quantization.

    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer.
    """
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size

    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"
    assert hidden_states.shape[1] == w1.shape[1] * 16, "Hidden size mismatch w1"
    assert hidden_states.shape[1] == w2.shape[2] // (
        num_bits // 2
    ), "Hidden size mismatch w2"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float16, torch.bfloat16]
    is_mxfp4_marlin = (
        num_bits == 4
        and w1_zeros is None
        and w2_zeros is None
        and w1_scale.dtype == torch.float8_e8m0fnu
        and w2_scale.dtype == torch.float8_e8m0fnu
    )
    if is_mxfp4_marlin:
        assert hidden_states.dtype == torch.bfloat16, (
            "MXFP4 Marlin with E8M0 scales is only instantiated for bfloat16 "
            f"activations, got {hidden_states.dtype}"
        )
    else:
        assert (
            hidden_states.dtype == w1_scale.dtype
        ), f"moe_wna16_marlin_gemm assumes hidden_states.dtype ({hidden_states.dtype}) == w1_scale.dtype ({w1_scale.dtype})"
        assert (
            hidden_states.dtype == w2_scale.dtype
        ), f"moe_wna16_marlin_gemm assumes hidden_states.dtype ({hidden_states.dtype}) == w2_scale.dtype ({w2_scale.dtype})"
    assert num_bits in [4, 8]

    M, K = hidden_states.shape
    E = w1.shape[0]
    N = w2.shape[1] * 16
    topk = topk_ids.shape[1]

    # M block size selection logic
    # TODO: tune this further for specific models
    for block_size_m in [8, 16, 32, 48, 64]:
        if M * topk / E / block_size_m < 0.9:
            break

    if global_num_experts == -1:
        global_num_experts = E
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, block_size_m, global_num_experts
    )

    if workspace is None:
        max_workspace_size = (max(2 * N, K) // 64) * (
            sorted_token_ids.size(0) // block_size_m
        )
        device = hidden_states.device
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        max_workspace_size = min(max_workspace_size, sms * 4)
        workspace = torch.zeros(
            max_workspace_size, dtype=torch.int, device=device, requires_grad=False
        )

    scalar_type1 = get_scalar_type(num_bits, w1_zeros is not None, w1_scale)
    scalar_type2 = get_scalar_type(num_bits, w2_zeros is not None, w2_scale)

    intermediate_cache2 = torch.empty(
        (M * topk_ids.shape[1], N),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache13 = torch.empty(
        (M * topk_ids.shape[1] * max(2 * N, K),),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache1 = intermediate_cache13[: M * topk_ids.shape[1] * 2 * N]
    intermediate_cache1 = intermediate_cache1.view(-1, 2 * N)
    intermediate_cache3 = intermediate_cache13[: M * topk_ids.shape[1] * K]
    intermediate_cache3 = intermediate_cache3.view(-1, K)

    use_atomic_add = (
        hidden_states.dtype == torch.half
        or torch.cuda.get_device_capability(hidden_states.device)[0] >= 9
    ) and (not is_mxfp4_marlin)

    intermediate_cache1 = moe_wna16_marlin_gemm(
        hidden_states,
        intermediate_cache1,
        w1,
        None,  # b_bias_or_none
        w1_scale,
        None,  # global_scale_or_none
        w1_zeros,
        g_idx1,
        sort_indices1,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        moe_block_size=block_size_m,
        top_k=topk,
        mul_topk_weights=False,
        is_ep=expert_map is not None,
        b_q_type=scalar_type1,
        size_m=M,
        size_n=2 * N,
        size_k=K,
        is_k_full=is_k_full,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=True,
        is_zp_float=False,
    )

    if clamp_limit is not None:
        swiglu_limit_func(
            intermediate_cache2,
            intermediate_cache1.view(-1, 2 * N),
            clamp_limit,
        )
    else:
        silu_and_mul(intermediate_cache1.view(-1, 2 * N), intermediate_cache2)

    if expert_map is not None:
        intermediate_cache3.zero_()

    intermediate_cache3 = moe_wna16_marlin_gemm(
        intermediate_cache2,
        intermediate_cache3,
        w2,
        None,  # b_bias_or_none
        w2_scale,
        None,  # global_scale_or_none
        w2_zeros,
        g_idx2,
        sort_indices2,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        moe_block_size=block_size_m,
        top_k=1,
        mul_topk_weights=True,
        is_ep=expert_map is not None,
        b_q_type=scalar_type2,
        size_m=M * topk,
        size_n=K,
        size_k=N,
        is_k_full=is_k_full,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=True,
        is_zp_float=False,
    ).view(-1, topk, K)

    output = hidden_states if inplace else torch.empty_like(hidden_states)

    if is_mxfp4_marlin:
        return torch.sum(intermediate_cache3, dim=1, out=output)
    else:
        if routed_scaling_factor is None:
            routed_scaling_factor = 1.0

        moe_sum_reduce(
            intermediate_cache3,
            output,
            routed_scaling_factor,
        )
        return output


@register_custom_op(out_shape="hidden_states")
def fused_marlin_moe_ep_packed(
    hidden_states: torch.Tensor,
    masked_m: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    g_idx1: Optional[torch.Tensor] = None,
    g_idx2: Optional[torch.Tensor] = None,
    sort_indices1: Optional[torch.Tensor] = None,
    sort_indices2: Optional[torch.Tensor] = None,
    w1_zeros: Optional[torch.Tensor] = None,
    w2_zeros: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    num_bits: int = 8,
    is_k_full: bool = True,
    clamp_limit: Optional[float] = None,
) -> torch.Tensor:
    """Run Marlin MoE on an EP-dispatched packed expert layout.

    ``hidden_states`` is expected to be laid out as
    [num_local_experts, max_tokens_per_expert, hidden_size]. ``masked_m`` gives
    the valid token count for each local expert; padded rows are ignored.
    """
    from sglang.srt.layers.moe.ep_moe.kernels import silu_and_mul_masked_fwd
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size

    assert hidden_states.ndim == 3, (
        "Packed EP Marlin expects hidden_states with shape "
        "[num_local_experts, max_tokens_per_expert, hidden_size], "
        f"got {tuple(hidden_states.shape)}"
    )
    assert hidden_states.is_contiguous(), "hidden_states must be contiguous"
    assert masked_m.ndim == 1, f"masked_m must be 1D, got {tuple(masked_m.shape)}"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float16, torch.bfloat16]

    num_local_experts, max_tokens_per_expert, K = hidden_states.shape
    assert masked_m.shape[0] == num_local_experts, (
        f"masked_m has {masked_m.shape[0]} experts, but hidden_states has "
        f"{num_local_experts}"
    )
    assert w1.shape[0] == num_local_experts, (
        f"w1 has {w1.shape[0]} experts, but hidden_states has {num_local_experts}"
    )
    assert w2.shape[0] == num_local_experts, (
        f"w2 has {w2.shape[0]} experts, but hidden_states has {num_local_experts}"
    )
    assert K == w1.shape[1] * 16, "Hidden size mismatch w1"
    assert K == w2.shape[2] // (num_bits // 2), "Hidden size mismatch w2"

    is_mxfp4_marlin = (
        num_bits == 4
        and w1_zeros is None
        and w2_zeros is None
        and w1_scale.dtype == torch.float8_e8m0fnu
        and w2_scale.dtype == torch.float8_e8m0fnu
    )
    if is_mxfp4_marlin:
        assert hidden_states.dtype == torch.bfloat16, (
            "MXFP4 Marlin with E8M0 scales is only instantiated for bfloat16 "
            f"activations, got {hidden_states.dtype}"
        )
    else:
        assert hidden_states.dtype == w1_scale.dtype, (
            f"moe_wna16_marlin_gemm assumes hidden_states.dtype "
            f"({hidden_states.dtype}) == w1_scale.dtype ({w1_scale.dtype})"
        )
        assert hidden_states.dtype == w2_scale.dtype, (
            f"moe_wna16_marlin_gemm assumes hidden_states.dtype "
            f"({hidden_states.dtype}) == w2_scale.dtype ({w2_scale.dtype})"
        )
    assert num_bits in [4, 8]

    M = num_local_experts * max_tokens_per_expert
    N = w2.shape[1] * 16
    topk = 1

    token_offsets = torch.arange(
        max_tokens_per_expert, device=hidden_states.device, dtype=masked_m.dtype
    ).unsqueeze(0)
    local_expert_ids = torch.arange(
        num_local_experts, device=hidden_states.device, dtype=masked_m.dtype
    ).unsqueeze(1)
    valid_token_mask = token_offsets < masked_m.unsqueeze(1)
    packed_topk_ids = torch.where(
        valid_token_mask,
        local_expert_ids.expand(num_local_experts, max_tokens_per_expert),
        torch.full(
            (num_local_experts, max_tokens_per_expert),
            -1,
            device=hidden_states.device,
            dtype=masked_m.dtype,
        ),
    ).reshape(-1, 1)

    # M block size selection logic mirrors fused_marlin_moe. Here M/E is the
    # per-expert packed capacity.
    for block_size_m in [8, 16, 32, 48, 64]:
        if M * topk / num_local_experts / block_size_m < 0.9:
            break

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        packed_topk_ids, block_size_m, num_local_experts
    )

    if workspace is None:
        max_workspace_size = (max(2 * N, K) // 64) * (
            sorted_token_ids.size(0) // block_size_m
        )
        device = hidden_states.device
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        max_workspace_size = min(max_workspace_size, sms * 4)
        workspace = torch.zeros(
            max_workspace_size, dtype=torch.int, device=device, requires_grad=False
        )

    scalar_type1 = get_scalar_type(num_bits, w1_zeros is not None, w1_scale)
    scalar_type2 = get_scalar_type(num_bits, w2_zeros is not None, w2_scale)

    flat_hidden_states = hidden_states.view(M, K)
    marlin_topk_weights = torch.ones(
        (M, 1), device=hidden_states.device, dtype=torch.float32
    )

    intermediate_cache1 = torch.empty(
        (M, 2 * N),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    use_atomic_add = (
        hidden_states.dtype == torch.half
        or torch.cuda.get_device_capability(hidden_states.device)[0] >= 9
    ) and (not is_mxfp4_marlin)

    intermediate_cache1 = moe_wna16_marlin_gemm(
        flat_hidden_states,
        intermediate_cache1,
        w1,
        None,  # b_bias_or_none
        w1_scale,
        None,  # global_scale_or_none
        w1_zeros,
        g_idx1,
        sort_indices1,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        marlin_topk_weights,
        moe_block_size=block_size_m,
        top_k=topk,
        mul_topk_weights=False,
        is_ep=True,
        b_q_type=scalar_type1,
        size_m=M,
        size_n=2 * N,
        size_k=K,
        is_k_full=is_k_full,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=True,
        is_zp_float=False,
    )

    intermediate_cache2 = torch.empty(
        (num_local_experts, max_tokens_per_expert, N),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    gateup = intermediate_cache1.view(num_local_experts, max_tokens_per_expert, 2 * N)
    if hidden_states.dtype == torch.bfloat16 and clamp_limit is None:
        silu_and_mul_masked_fwd(gateup, intermediate_cache2, masked_m)
    else:
        gateup.masked_fill_(~valid_token_mask.unsqueeze(-1), 0)
        if clamp_limit is not None:
            swiglu_limit_func(
                intermediate_cache2.view(M, N),
                gateup.view(M, 2 * N),
                clamp_limit,
            )
        else:
            silu_and_mul(gateup.view(M, 2 * N), intermediate_cache2.view(M, N))

    output = torch.empty(
        (M, K),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    output = moe_wna16_marlin_gemm(
        intermediate_cache2.view(M, N),
        output,
        w2,
        None,  # b_bias_or_none
        w2_scale,
        None,  # global_scale_or_none
        w2_zeros,
        g_idx2,
        sort_indices2,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        marlin_topk_weights,
        moe_block_size=block_size_m,
        top_k=topk,
        mul_topk_weights=False,
        is_ep=True,
        b_q_type=scalar_type2,
        size_m=M,
        size_n=K,
        size_k=N,
        is_k_full=is_k_full,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=True,
        is_zp_float=False,
    )

    return output.view(num_local_experts, max_tokens_per_expert, K)
