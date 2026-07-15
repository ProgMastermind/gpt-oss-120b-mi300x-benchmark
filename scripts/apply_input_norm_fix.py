#!/usr/bin/env python3
"""
apply_input_norm_fix.py — Port the input_norm / norm_before_fc fix from our
Eagle3 PR to ATOM v0.1.6-rc0's eagle3_llama.py.

Run on the droplet AFTER installing ATOM v0.1.6-rc0:
    python3 apply_input_norm_fix.py /root/ATOM/atom/models/eagle3_llama.py

Idempotent: if the fix is already present, exits 0 with a message.
"""
import sys
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 apply_input_norm_fix.py <path/to/eagle3_llama.py>")
        sys.exit(1)

    fpath = Path(sys.argv[1])
    if not fpath.exists():
        print(f"ERROR: {fpath} not found")
        sys.exit(1)

    src = fpath.read_text()

    # Idempotency check
    if "input_norm" in src:
        print(f"input_norm fix already present in {fpath}. Nothing to do.")
        return

    # --- Edit 1: add norm_before_fc attribute after norm_output ---
    old1 = "        self.norm_output = getattr(config, \"norm_output\", False)\n"
    new1 = (
        "        self.norm_output = getattr(config, \"norm_output\", False)\n"
        "        self.norm_before_fc = getattr(config, \"norm_before_fc\", False)\n"
    )
    if old1 not in src:
        print("ERROR: anchor 1 (norm_output) not found. File may have changed.")
        sys.exit(1)
    src = src.replace(old1, new1, 1)

    # --- Edit 2: add input_norm module after fc_norm block ---
    old2 = "        else:\n            self.fc_norm = None\n\n        # Draft attention layer_num"
    new2 = (
        "        else:\n"
        "            self.fc_norm = None\n"
        "\n"
        "        if self.norm_before_fc:\n"
        "            self.input_norm = RMSNorm(\n"
        "                target_hidden_size * num_aux, eps=config.rms_norm_eps\n"
        "            )\n"
        "        else:\n"
        "            self.input_norm = None\n"
        "\n"
        "        # Draft attention layer_num"
    )
    if old2 not in src:
        print("ERROR: anchor 2 (fc_norm None block) not found.")
        sys.exit(1)
    src = src.replace(old2, new2, 1)

    # --- Edit 3: apply input_norm in fc_norm is None path ---
    old3 = (
        "            else:\n"
        "                fc_in = aux_hidden_states\n"
        "            return self.fc(fc_in)\n"
    )
    new3 = (
        "            else:\n"
        "                fc_in = aux_hidden_states\n"
        "            if self.input_norm is not None:\n"
        "                fc_in = self.input_norm(fc_in)\n"
        "            return self.fc(fc_in)\n"
    )
    if old3 not in src:
        print("ERROR: anchor 3 (fc_norm None return) not found.")
        sys.exit(1)
    src = src.replace(old3, new3, 1)

    # --- Edit 4: apply input_norm in fc_norm path (before fused_group_rmsnorm) ---
    old4 = (
        "        x = torch.cat(aux_hidden_states, dim=-1) if is_list else aux_hidden_states\n"
        "        if (\n"
    )
    new4 = (
        "        x = torch.cat(aux_hidden_states, dim=-1) if is_list else aux_hidden_states\n"
        "        if self.input_norm is not None:\n"
        "            x = self.input_norm(x)\n"
        "        if (\n"
    )
    if old4 not in src:
        print("ERROR: anchor 4 (fc_norm path x = torch.cat) not found.")
        sys.exit(1)
    src = src.replace(old4, new4, 1)

    fpath.write_text(src)
    print(f"SUCCESS: input_norm / norm_before_fc fix applied to {fpath}")

    # Verify
    verify = fpath.read_text()
    assert "self.norm_before_fc" in verify, "norm_before_fc missing after edit"
    assert "self.input_norm = RMSNorm" in verify, "input_norm init missing"
    assert "fc_in = self.input_norm(fc_in)" in verify, "input_norm apply (path 1) missing"
    assert "x = self.input_norm(x)" in verify, "input_norm apply (path 2) missing"
    print("VERIFIED: all 4 edits present.")


if __name__ == "__main__":
    main()
