#!/bin/bash

# Try to power off the VM
echo "Shutting down..."
vboxmanage controlvm blockplayer poweroff

# Restore from a snapshot "Fresh SSH" which must have already been created.
# "Fresh SSH" should be a clean installation of Ubuntu with
#    user: blockplayer
#    password: pass
# SSH server must be running, 
#    apt-get install openssh-server
# 

set -e
vboxmanage snapshot blockplayer restore "Fresh SSH"

# Configure port forwarding so we can connect to this device without knowing its IP
# You can connect to the virtual machine using blockplayer@localhost -p2222.
VBoxManage setextradata blockplayer "VBoxInternal/Devices/e1000/0/LUN#0/Config/ssh/HostPort" 2222
VBoxManage setextradata blockplayer "VBoxInternal/Devices/e1000/0/LUN#0/Config/ssh/GuestPort" 22
VBoxManage setextradata blockplayer "VBoxInternal/Devices/e1000/0/LUN#0/Config/ssh/Protocol" TCP

VBoxManage setextradata blockplayer "VBoxInternal/Devices/e1000/0/LUN#0/Config/lighttpd/HostPort" 8090
VBoxManage setextradata blockplayer "VBoxInternal/Devices/e1000/0/LUN#0/Config/lighttpd/GuestPort" 8090
VBoxManage setextradata blockplayer "VBoxInternal/Devices/e1000/0/LUN#0/Config/lighttpd/Protocol" TCP


# Start the VM in headless mode
vboxmanage startvm blockplayer --type headless

# Give the VM time to start up by pinging it via the forwarded port
echo "Pinging..."
until sshpass -p pass ssh -p2222 blockplayer@localhost "echo ok"; do sleep 1; done

# Run the post install script
# NOTE this requires sshpass-1.05-1, which is not in the ubuntu repository 
# yet as of Feb 23, 2012
sshpass -p pass scp -P2222 vmdist/install_vm.sh blockplayer@localhost:
sshpass -p pass ssh -p2222 -tt blockplayer@localhost "bash install_vm.sh"
