# vLLM — Qwen3-32B FP16, 4x L20, TP=4
# Usage: terraform apply -var-file=configs/vllm-4xl20.tfvars

instance_name       = "vllm-qwen3-32b-4xL20"
instance_type       = "PNV5b.48XLARGE768"   # 4x L20 48GB, 192 vCPU, 768GB RAM
data_disk_size      = 500
security_group_name = "vllm-4xl20-sg"
