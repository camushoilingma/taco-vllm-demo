output "instance_id" {
  description = "CVM instance ID"
  value       = tencentcloud_instance.gpu.id
}

output "instance_ip" {
  description = "Public IP address"
  value       = tencentcloud_instance.gpu.public_ip
}

output "ssh_command" {
  description = "SSH command to connect"
  value       = "ssh -i <your-ssh-key> ubuntu@${tencentcloud_instance.gpu.public_ip}"
}

output "security_group_id" {
  description = "Security group ID"
  value       = tencentcloud_security_group.demo.id
}
