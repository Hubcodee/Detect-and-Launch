provider "aws" {
  region  = "ap-southeast-1"
  profile = "tf-user"
}

resource "aws_instance" "demo_deployment" {
  ami           = "ami-0f511ead81ccde020"
  instance_type = "t2.micro"
  tags = {
    Name    = "Test"
    Service = "demo"
  }

}
output "os_value" {
  value = aws_instance.demo_deployment
}
output "public_ip" {
  value = aws_instance.demo_deployment.public_ip
}
output "az" {
  value = aws_instance.demo_deployment.availability_zone
}

resource "aws_ebs_volume" "eb" {
  availability_zone = aws_instance.demo_deployment.availability_zone
  size              = 10
  tags = {
    Name : "Test_EB"
  }
}
output "id" {
  value = aws_ebs_volume.eb
}

resource "aws_volume_attachment" "eb_attach" {
  device_name = "/dev/xvdf"
  volume_id   = aws_ebs_volume.eb.id
  instance_id = aws_instance.demo_deployment.id
}


