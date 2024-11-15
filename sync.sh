#!/bin/bash
rsync -av -e "ssh -i ~/.ssh/id_rsa_jai" --info=progress2 --exclude={.git,datasets,ckpts,__pycache__,uqag_pt,sync.sh,env,out} ./ "jai.bhatnagar@ada:/home2/jai.bhatnagar/uqag"
# 
