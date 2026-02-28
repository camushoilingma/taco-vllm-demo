terraform {
  required_providers {
    tencentcloud = {
      source  = "tencentcloudstack/tencentcloud"
      version = ">= 1.81.0"
    }
  }
  required_version = ">= 1.0"
}

provider "tencentcloud" {
  region     = var.region
  secret_id  = var.secret_id
  secret_key = var.secret_key
}

# ── Security Group ──

resource "tencentcloud_security_group" "demo" {
  name        = var.security_group_name
  description = "Security group for TACO/vLLM benchmark instance"
}

resource "tencentcloud_security_group_lite_rule" "demo" {
  security_group_id = tencentcloud_security_group.demo.id

  ingress = [
    "ACCEPT#${var.my_ip}#22#TCP",       # SSH
    "ACCEPT#${var.my_ip}#8000#TCP",     # vLLM API
    "ACCEPT#${var.my_ip}#18080#TCP",    # TACO-X API
  ]

  egress = [
    "ACCEPT#0.0.0.0/0#ALL#ALL",
  ]
}

# ── GPU Instance ──

resource "tencentcloud_instance" "gpu" {
  instance_name              = var.instance_name
  instance_type              = var.instance_type
  image_id                   = var.image_id
  availability_zone          = var.availability_zone
  disable_api_termination    = var.disable_terminate

  vpc_id                     = var.vpc_id
  subnet_id                  = var.subnet_id
  allocate_public_ip         = true
  internet_max_bandwidth_out = var.bandwidth
  internet_charge_type       = "TRAFFIC_POSTPAID_BY_HOUR"

  orderly_security_groups    = [tencentcloud_security_group.demo.id]

  key_ids                    = [var.key_id]

  system_disk_type = "CLOUD_SSD"
  system_disk_size = var.system_disk_size

  data_disks {
    data_disk_type = "CLOUD_SSD"
    data_disk_size = var.data_disk_size
  }

  tags = var.tags
}
