#!/bin/bash

vboxmanage controlvm blockplayer poweroff

VBOX_IP=192.168.1.110

set -e
vboxmanage snapshot blockplayer restore "Fresh SSH"
vboxmanage startvm blockplayer --type headless

until ping -qc 1 ${VBOX_IP}; do sleep 1; done

sshpass -p user scp vmdist/install_vm.sh user@${VBOX_IP}:
sshpass -p user ssh -tt -A user@${VBOX_IP} "bash install_vm.sh"
