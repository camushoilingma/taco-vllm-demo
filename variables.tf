## ── Tencent Cloud Credentials ──

variable "secret_id" {
  description = "Tencent Cloud API Secret ID"
  type        = string
  sensitive   = true
}

variable "secret_key" {
  description = "Tencent Cloud API Secret Key"
  type        = string
  sensitive   = true
}

## ── Region & Availability Zone ──

variable "region" {
  type    = string
  default = "ap-tokyo"
}

variable "availability_zone" {
  type    = string
  default = "ap-tokyo-2"
}

## ── Instance Configuration ──

variable "instance_name" {
  type    = string
  default = "taco-benchmark"
}

variable "instance_type" {
  description = <<-EOT
    GPU instance type. Options:
      PNV5b.8XLARGE96    = 1x L20 48GB,  32 vCPU,  96GB RAM
      PNV5b.16XLARGE192  = 2x L20 48GB,  64 vCPU, 192GB RAM
      PNV5b.48XLARGE768  = 4x L20 48GB, 192 vCPU, 768GB RAM
  EOT
  type    = string
  default = "PNV5b.16XLARGE192"
}

variable "image_id" {
  description = "Ubuntu 22.04 LTS (img-487zeit5 in Tokyo)"
  type        = string
  default     = "img-487zeit5"
}

variable "system_disk_size" {
  type    = number
  default = 100
}

variable "data_disk_size" {
  description = "Data disk in GB. 200 for 2xL20, 500 for 4xL20 (model + docker image)"
  type        = number
  default     = 200
}

## ── Network ──

variable "vpc_id"    { type = string }
variable "subnet_id" { type = string }

variable "bandwidth" {
  type    = number
  default = 150
}

## ── SSH & Security ──

variable "key_id" {
  type    = string
  default = "skey-xxxxxxxx"
}

variable "my_ip" {
  description = "Your public IP in CIDR (e.g. 81.249.144.139/32)"
  type        = string
}

variable "security_group_name" {
  type    = string
  default = "taco-benchmark-sg"
}

## ── Tags ──

variable "tags" {
  type = map(string)
  default = {
    TaggerOwner     = "your-name"
    TaggerProject   = "taco-x-benchmark"
    TaggerTTL       = "7"
    TaggerCanDelete = "YES"
    TaggerAutoOff   = "YES"
    TaggerAutoStart = "NO"
  }
}

## ── Protection ──

variable "disable_terminate" {
  type    = bool
  default = false
}
