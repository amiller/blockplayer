#!/bin/bash

vboxmanage controlvm blockplayer poweroff



set -e
vboxmanage snapshot blockplayer restore "Fresh SSH"

VBoxManage setextradata blockplayer "VBoxInternal/Devices/e1000/0/LUN#0/Config/ssh/HostPort" 2222
VBoxManage setextradata blockplayer "VBoxInternal/Devices/e1000/0/LUN#0/Config/ssh/GuestPort" 22
VBoxManage setextradata blockplayer "VBoxInternal/Devices/e1000/0/LUN#0/Config/ssh/Protocol" TCP

#vboxmanage startvm blockplayer --type headless


#until ping -qc 1 ${VBOX_IP}; do sleep 1; done

#sshpass -p user scp vmdist/install_vm.sh user@${VBOX_IP}:
#sshpass -p user ssh -tt -A user@${VBOX_IP} "bash install_vm.sh"
