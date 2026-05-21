from __future__ import annotations

import torch


class CESTASequenceMixin:
    crf_transitions: torch.Tensor

    def crf_negative_log_likelihood(
        self,
        emissions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        emissions_seq, targets_seq, mask_seq = self._crf_sequence_tensors(emissions, targets, mask)
        has_valid = mask_seq.any(dim=1)
        if not bool(has_valid.any()):
            return emissions.sum() * 0.0
        emissions_seq = emissions_seq[has_valid]
        targets_seq = targets_seq[has_valid]
        mask_seq = mask_seq[has_valid]
        safe_targets = targets_seq.clamp_min(0)

        gathered = emissions_seq.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
        first_emission = gathered[:, 0] * mask_seq[:, 0].to(emissions_seq.dtype)
        transition_scores = self.crf_transitions[safe_targets[:, :-1], safe_targets[:, 1:]]
        transition_mask = mask_seq[:, :-1] & mask_seq[:, 1:]
        sequence_score = first_emission + ((transition_scores + gathered[:, 1:]) * transition_mask.to(emissions_seq.dtype)).sum(dim=1)

        alpha = emissions_seq[:, 0]
        alpha = torch.where(mask_seq[:, 0].unsqueeze(-1), alpha, torch.zeros_like(alpha))
        for timestep in range(1, emissions_seq.size(1)):
            scores = alpha.unsqueeze(2) + self.crf_transitions.unsqueeze(0) + emissions_seq[:, timestep].unsqueeze(1)
            next_alpha = torch.logsumexp(scores, dim=1)
            alpha = torch.where(mask_seq[:, timestep].unsqueeze(-1), next_alpha, alpha)
        log_partition = torch.logsumexp(alpha, dim=1)
        return (log_partition - sequence_score).mean()

    @torch.no_grad()
    def crf_decode(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        original_shape = emissions.shape[:-1]
        emissions_seq, _, mask_seq = self._crf_sequence_tensors(emissions, None, mask)
        if emissions_seq.numel() == 0:
            return emissions.argmax(dim=-1)
        score = emissions_seq[:, 0]
        score = torch.where(mask_seq[:, 0].unsqueeze(-1), score, torch.zeros_like(score))
        backpointers: list[torch.Tensor] = []
        for timestep in range(1, emissions_seq.size(1)):
            transition_score = score.unsqueeze(2) + self.crf_transitions.unsqueeze(0)
            best_score, best_path = transition_score.max(dim=1)
            next_score = best_score + emissions_seq[:, timestep]
            score = torch.where(mask_seq[:, timestep].unsqueeze(-1), next_score, score)
            backpointers.append(best_path)

        best_last = score.argmax(dim=1)
        decoded = emissions_seq.new_zeros(emissions_seq.shape[:2], dtype=torch.long)
        decoded[:, -1] = best_last
        for reverse_index, backpointer in enumerate(reversed(backpointers), start=1):
            timestep = emissions_seq.size(1) - reverse_index
            previous = backpointer.gather(1, decoded[:, timestep].unsqueeze(1)).squeeze(1)
            decoded[:, timestep - 1] = torch.where(mask_seq[:, timestep], previous, decoded[:, timestep])
        decoded = decoded.masked_fill(~mask_seq, 0)
        if len(original_shape) == 3:
            batch, seq_len, num_nodes = original_shape
            return decoded.view(batch, num_nodes, seq_len).permute(0, 2, 1)
        return decoded.view(*original_shape)

    def _crf_sequence_tensors(
        self,
        emissions: torch.Tensor,
        targets: torch.Tensor | None,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if emissions.ndim == 4:
            batch, seq_len, num_nodes, num_classes = emissions.shape
            emissions_seq = emissions.permute(0, 2, 1, 3).reshape(batch * num_nodes, seq_len, num_classes)
            if targets is None:
                targets_seq = torch.zeros(batch * num_nodes, seq_len, dtype=torch.long, device=emissions.device)
            else:
                targets_seq = targets.permute(0, 2, 1).reshape(batch * num_nodes, seq_len)
            if mask is None:
                mask_seq = (
                    targets_seq >= 0
                    if targets is not None
                    else torch.ones(batch * num_nodes, seq_len, dtype=torch.bool, device=emissions.device)
                )
            else:
                mask_seq = mask.permute(0, 2, 1).reshape(batch * num_nodes, seq_len)
        elif emissions.ndim == 3:
            batch, seq_len, num_classes = emissions.shape
            emissions_seq = emissions.reshape(batch, seq_len, num_classes)
            if targets is None:
                targets_seq = torch.zeros(batch, seq_len, dtype=torch.long, device=emissions.device)
            else:
                targets_seq = targets.reshape(batch, seq_len)
            if mask is None:
                mask_seq = (
                    targets_seq >= 0
                    if targets is not None
                    else torch.ones(batch, seq_len, dtype=torch.bool, device=emissions.device)
                )
            else:
                mask_seq = mask.reshape(batch, seq_len)
        else:
            raise ValueError("CRF emissions must have shape (B, T, C) or (B, T, N, C)")
        if targets is not None:
            mask_seq = mask_seq & (targets_seq >= 0)
        return emissions_seq, targets_seq.long(), mask_seq.bool()
