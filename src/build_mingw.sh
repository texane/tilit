#!/usr/bin/env sh
gcc \
    -Iopencv/include \
    -Wno-unused-function \
    -Wno-implicit-function-declaration \
    -Wall -O3 main.c \
    -Lopencv/lib -lopencv_world341

#    -Lopencv/bin -lopencv_ffmpeg341


