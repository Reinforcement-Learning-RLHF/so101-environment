"""
Push ACTPolicy to Hugging Face Hub (Public)

This script uploads your trained ACTPolicy to a public Hugging Face repository,
so anyone can pull it later using `ACTPolicy.from_pretrained("username/repo_name")`.
"""

from lerobot.policies.act.modeling_act import ACTPolicy

# Path where your policy is saved locally
LOCAL_POLICY_PATH = "policies/50k_policy"

# Hugging Face repository
REPO_ID = "Ishah8840/so101_act_policy"

# Load your trained policy
policy = ACTPolicy.from_pretrained(LOCAL_POLICY_PATH)

# Push to HF Hub
policy.push_to_hub(REPO_ID)
print(f"Policy successfully uploaded to Hugging Face Hub: {REPO_ID}")