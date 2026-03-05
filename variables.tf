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
      PNV5b.48XLARGE768  = 4x L20 48GB, 192 vCPU, 768GB RAM
  EOT
  type    = string
  default = "PNV5b.48XLARGE768"
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
  description = "Data disk in GB. 500 for 4xL20 (model + docker image)"
  type        = number
  default     = 500
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
  default = "skey-1x6dw3nj"
}

variable "security_group_id" {
  description = "Existing security group ID to attach"
  type        = string
  default     = "sg-glu6qmn6"
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
