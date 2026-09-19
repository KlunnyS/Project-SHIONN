#!/usr/bin/env bash

cd "/home/user/.local/share/Steam/steamapps/common/Portal 2/bin" || exit 1

unset __NV_PRIME_RENDER_OFFLOAD
unset __GLX_VENDOR_LIBRARY_NAME
unset __VK_LAYER_NV_optimus

exec env \
    DRI_PRIME=0 \
    wine ./hammerplusplus.exe
