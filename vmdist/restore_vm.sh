#!/bin/bash

vboxmanage controlvm blockplayer poweroff

set -e
vboxmanage snapshot blockplayer restore "Fresh SSH"
vboxmanage startvm blockplayer --type headless

until ping -qc 1 192.168.1.112; do sleep 1; done

sshpass -p user scp vmdist/install_vm.sh user@192.168.1.112:
sshpass -p user ssh -tt -A user@192.168.1.112 "bash install_vm.sh"

'
