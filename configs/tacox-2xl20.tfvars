# TACO-X — Qwen3-32B FP16, 2x L20, TP=2
# Usage: terraform apply -var-file=configs/tacox-2xl20.tfvars

instance_name       = "tacox-qwen3-32b-2xL20"
instance_type       = "PNV5b.16XLARGE192"   # 2x L20 48GB, 64 vCPU, 192GB RAM
data_disk_size      = 200
security_group_name = "tacox-2xl20-sg"
